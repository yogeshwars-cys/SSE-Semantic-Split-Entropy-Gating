// sse.hpp: C++ port of the SSE prototype hybrid, fully binary semantic store with int8 rescore (HYB_QUANT=3).
//
// Mirrors, step for step, sse/anomaly_model.py (statistical gate), harness/quant_space.py (CompressedSpace with
// binary=True, int8=True), harness/replay_hybrid2.py (TokenThresholds packed + int8 + portable, Hybrid2 "full") and
// the request mapping of harness/run_sse_srbh.py (to_request with request fields). Floating point follows the Python
// order of operations (Neumaier sums where Python's sum() is used, sequential sums where numpy cumsum is used,
// exact-rational statistics approximated in __float128), so a replay reproduces the Python verdicts request by
// request. Build with -ffp-contract=off: a fused multiply-add would change the last bit of a score.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cwctype>
#include <quadmath.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sse {

constexpr int DIM = 4096;

// ------------------------------------------------------------------------------------------------ text helpers

inline uint32_t crc32_update(uint32_t c, const uint8_t* p, size_t n) {   // raw register: no pre/post inversion
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    for (size_t i = 0; i < n; i++) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c;
}
inline uint32_t crc32(const uint8_t* p, size_t n) { return crc32_update(0xFFFFFFFFu, p, n) ^ 0xFFFFFFFFu; }
inline uint32_t crc32(std::string_view s) { return crc32(reinterpret_cast<const uint8_t*>(s.data()), s.size()); }

inline void put_utf8(std::string& o, uint32_t c) {
    if (c < 0x80) o += char(c);
    else if (c < 0x800) { o += char(0xC0 | (c >> 6)); o += char(0x80 | (c & 63)); }
    else if (c < 0x10000) { o += char(0xE0 | (c >> 12)); o += char(0x80 | ((c >> 6) & 63)); o += char(0x80 | (c & 63)); }
    else { o += char(0xF0 | (c >> 18)); o += char(0x80 | ((c >> 12) & 63)); o += char(0x80 | ((c >> 6) & 63)); o += char(0x80 | (c & 63)); }
}

// bytes -> code points, CPython's utf-8 decoder with errors="replace" (one U+FFFD per maximal invalid subpart)
inline void decode_utf8(std::string_view s, std::vector<uint32_t>& out) {
    out.clear();
    const auto* b = reinterpret_cast<const uint8_t*>(s.data());
    size_t n = s.size(), i = 0;
    auto cont = [&](size_t j) { return j < n && (b[j] & 0xC0) == 0x80; };
    while (i < n) {
        uint8_t c = b[i];
        if (c < 0x80) { out.push_back(c); i++; continue; }
        if (c < 0xC2 || c > 0xF4) { out.push_back(0xFFFD); i++; continue; }
        if (c < 0xE0) {
            if (cont(i + 1)) { out.push_back(((c & 31u) << 6) | (b[i + 1] & 63u)); i += 2; }
            else { out.push_back(0xFFFD); i++; }
            continue;
        }
        uint8_t lo = 0x80, hi = 0xBF;
        if (c == 0xE0) lo = 0xA0; else if (c == 0xED) hi = 0x9F; else if (c == 0xF0) lo = 0x90; else if (c == 0xF4) hi = 0x8F;
        if (!(i + 1 < n && b[i + 1] >= lo && b[i + 1] <= hi)) { out.push_back(0xFFFD); i++; continue; }
        if (c < 0xF0) {
            if (cont(i + 2)) { out.push_back(((c & 15u) << 12) | ((b[i + 1] & 63u) << 6) | (b[i + 2] & 63u)); i += 3; }
            else { out.push_back(0xFFFD); i += 2; }
            continue;
        }
        if (!cont(i + 2)) { out.push_back(0xFFFD); i += 2; continue; }
        if (!cont(i + 3)) { out.push_back(0xFFFD); i += 3; continue; }
        out.push_back(((c & 7u) << 18) | ((b[i + 1] & 63u) << 12) | ((b[i + 2] & 63u) << 6) | (b[i + 3] & 63u));
        i += 4;
    }
}

inline std::string sanitize_utf8(std::string_view s) {
    bool ascii = true;
    for (char c : s) if (uint8_t(c) >= 0x80) { ascii = false; break; }
    if (ascii) return std::string(s);
    std::vector<uint32_t> cp;
    decode_utf8(s, cp);
    std::string o;
    o.reserve(s.size());
    for (uint32_t c : cp) put_utf8(o, c);
    return o;
}

inline uint32_t lower_cp(uint32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    return uint32_t(std::towlower(wint_t(c)));                   // needs setlocale(LC_CTYPE, "C.UTF-8")
}

inline std::string lower_utf8(std::string_view s) {
    std::vector<uint32_t> cp;
    decode_utf8(s, cp);
    std::string o;
    for (uint32_t c : cp) put_utf8(o, lower_cp(c));
    return o;
}

// urllib.parse.unquote(errors="replace"): percent-decoding within ASCII runs only, each run decoded as utf-8
inline std::string unquote(std::string_view s, bool plus = false) {
    if (s.find('%') == std::string_view::npos) {
        if (!plus) return std::string(s);
        std::string o(s);
        std::replace(o.begin(), o.end(), '+', ' ');
        return o;
    }
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    std::string o, run;
    size_t i = 0, n = s.size();
    while (i < n) {
        if (uint8_t(s[i]) >= 0x80) { o += s[i++]; continue; }
        run.clear();
        while (i < n && uint8_t(s[i]) < 0x80) {
            char c = s[i];
            if (plus && c == '+') c = ' ';
            if (c == '%' && i + 2 < n) {                         // only %XX with two hex digits decodes
                int a = hex(s[i + 1]), b = hex(s[i + 2]);
                if (a >= 0 && b >= 0) { run += char(a * 16 + b); i += 3; continue; }
            }
            run += c;
            i++;
        }
        o += sanitize_utf8(run);
    }
    return o;
}

struct Url { std::string path, query; };

// urllib.parse.urlparse (Python 3.13): scheme, netloc, ;params of the last segment, #fragment and ?query split off
inline Url urlparse(std::string_view in) {
    size_t b = 0;
    while (b < in.size() && uint8_t(in[b]) <= 0x20) b++;
    std::string url;
    url.reserve(in.size() - b);
    for (size_t i = b; i < in.size(); i++)
        if (in[i] != '\t' && in[i] != '\r' && in[i] != '\n') url += in[i];
    std::string scheme;
    size_t c = url.find(':');
    if (c != std::string::npos && c > 0 && std::isalpha(uint8_t(url[0])) && uint8_t(url[0]) < 0x80) {
        bool ok = true;
        for (size_t i = 0; i < c; i++) {
            uint8_t ch = url[i];
            if (!(ch < 0x80 && (std::isalnum(ch) || ch == '+' || ch == '-' || ch == '.'))) { ok = false; break; }
        }
        if (ok) {
            scheme = url.substr(0, c);
            for (auto& ch : scheme) ch = char(std::tolower(uint8_t(ch)));
            url = url.substr(c + 1);
        }
    }
    if (url.compare(0, 2, "//") == 0) {
        size_t d = url.size();
        for (char ch : {'/', '?', '#'}) {
            size_t w = url.find(ch, 2);
            if (w != std::string::npos) d = std::min(d, w);
        }
        url = url.substr(d);
    }
    size_t h = url.find('#');
    if (h != std::string::npos) url.resize(h);
    Url r;
    size_t q = url.find('?');
    if (q != std::string::npos) { r.query = url.substr(q + 1); url.resize(q); }
    static const char* uses_params[] = {"", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp",
                                        "rtsps", "rtspu", "sip", "sips", "mms", "sftp", "tel"};
    bool params = false;
    for (auto* u : uses_params) if (scheme == u) { params = true; break; }
    if (params && url.find(';') != std::string::npos) {
        size_t sl = url.rfind('/'), i = sl == std::string::npos ? url.find(';') : url.find(';', sl);
        if (i != std::string::npos) url.resize(i);
    }
    r.path = url;
    return r;
}

// urllib.parse.parse_qs(keep_blank_values=True) keys, in first-seen order
inline void parse_qs_keys(std::string_view qs, std::vector<std::string>& out) {
    if (qs.empty()) return;
    std::vector<std::string> mine;
    size_t s = 0;
    while (s <= qs.size()) {
        size_t e = qs.find('&', s);
        if (e == std::string_view::npos) e = qs.size();
        std::string_view nv = qs.substr(s, e - s);
        if (!nv.empty()) {
            size_t eq = nv.find('=');
            std::string name = unquote(eq == std::string_view::npos ? nv : nv.substr(0, eq), true);
            if (std::find(mine.begin(), mine.end(), name) == mine.end()) mine.push_back(std::move(name));
        }
        s = e + 1;
    }
    for (auto& m : mine) out.push_back(std::move(m));
}

// ------------------------------------------------------------------------------------------------ numerics

struct NSum {                                                    // CPython 3.12+ sum() of floats (Neumaier)
    double f = 0, c = 0;
    void add(double x) {
        double t = f + x;
        if (std::fabs(f) >= std::fabs(x)) c += (f - t) + x; else c += (x - t) + f;
        f = t;
    }
    double get() const { return (c != 0 && std::isfinite(c)) ? f + c : f; }
};

inline double exact_mean(const std::vector<double>& x) {         // statistics.mean (exact rational, then rounded)
    __float128 s = 0;
    for (double v : x) s += v;
    return double(s / x.size());
}

inline double exact_pstdev(const std::vector<double>& x) {       // statistics.pstdev
    __float128 s = 0;
    for (double v : x) s += v;
    __float128 m = s / x.size(), ss = 0;
    for (double v : x) { __float128 d = v - m; ss += d * d; }
    return double(sqrtq(ss / x.size()));
}

struct SplitMix64 {                                              // quant_space.splitmix64
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return double(next() >> 11) * 0x1.0p-53; }
};

// ------------------------------------------------------------------------------------------------ requests

enum Field { UA, REFERER, HOST, ORIGIN, COOKIE, CTYPE, ACCEPT, ALANG, AENC, DNT, CONN, METHOD, PATH, QUERY, BODY,
             NFIELD, REQ = NFIELD };
inline const char* FIELD_NAMES[] = {"user-agent", "referer", "host", "origin", "cookie", "content-type", "accept",
                                    "accept-language", "accept-encoding", "dnt", "connection", "@method", "@path",
                                    "@query", "@body", "_req"};

struct Value {                                                   // a field value with what the gates read of it
    std::string s;                                               // utf-8
    int len = 0;                                                 // code points
    double ent = 0;                                              // shannon entropy over code points
    std::vector<int> gi, gc;                                     // sorted 3-gram ids, their counts
};

inline uint32_t crc32_update(uint32_t c, const uint8_t* p, size_t n);

inline void analyse_uncached(Value& v) {
    static thread_local std::vector<uint32_t> cp;
    decode_utf8(v.s, cp);
    v.len = int(cp.size());
    v.ent = 0;
    if (!cp.empty()) {                                           // anomaly_model.shannon: Counter order = first seen
        static thread_local std::vector<std::pair<uint32_t, int>> cnt;
        cnt.clear();
        int at[128];
        std::fill(at, at + 128, -1);
        for (uint32_t c : cp) {
            if (c < 128) {
                if (at[c] < 0) { at[c] = int(cnt.size()); cnt.push_back({c, 0}); }
                cnt[at[c]].second++;
            } else {
                size_t j = 0;
                while (j < cnt.size() && cnt[j].first != c) j++;
                if (j == cnt.size()) cnt.push_back({c, 0});
                cnt[j].second++;
            }
        }
        NSum s;
        double n = double(cp.size());
        for (auto& [c, k] : cnt) s.add(double(k) / n * std::log2(double(k) / n));
        v.ent = -s.get();
    }
    static thread_local std::vector<uint32_t> g;                 // semantic_model.grams
    g.clear();
    g.push_back(2);
    for (uint32_t c : cp) { c = lower_cp(c); g.push_back(c >= '0' && c <= '9' ? '0' : c); }
    g.push_back(3);
    static thread_local std::vector<uint16_t> cntg(DIM, 0);
    static thread_local std::vector<int> touched;
    touched.clear();
    for (size_t i = 0; i + 2 < g.size(); i++) {
        uint8_t buf[12];
        size_t n = 0;
        for (int k = 0; k < 3; k++) {                            // utf-8 of three code points
            uint32_t c = g[i + k];
            if (c < 0x80) buf[n++] = uint8_t(c);
            else if (c < 0x800) { buf[n++] = uint8_t(0xC0 | (c >> 6)); buf[n++] = uint8_t(0x80 | (c & 63)); }
            else if (c < 0x10000) { buf[n++] = uint8_t(0xE0 | (c >> 12)); buf[n++] = uint8_t(0x80 | ((c >> 6) & 63)); buf[n++] = uint8_t(0x80 | (c & 63)); }
            else { buf[n++] = uint8_t(0xF0 | (c >> 18)); buf[n++] = uint8_t(0x80 | ((c >> 12) & 63)); buf[n++] = uint8_t(0x80 | ((c >> 6) & 63)); buf[n++] = uint8_t(0x80 | (c & 63)); }
        }
        int id = int((crc32_update(0xFFFFFFFFu, buf, n) ^ 0xFFFFFFFFu) % DIM);
        if (!cntg[id]++) touched.push_back(id);
    }
    if (touched.empty()) { touched.push_back(0); cntg[0] = 1; }
    std::sort(touched.begin(), touched.end());
    v.gi.assign(touched.begin(), touched.end());
    v.gc.resize(touched.size());
    for (size_t i = 0; i < touched.size(); i++) { v.gc[i] = cntg[touched[i]]; cntg[touched[i]] = 0; }
}

// values repeat a lot (anomaly_model.shannon and semantic_model.gram_counts are lru_cached in Python too)
inline void analyse(Value& v) {
    static thread_local std::unordered_map<std::string, Value> cache;
    auto it = cache.find(v.s);
    if (it != cache.end()) { v.len = it->second.len; v.ent = it->second.ent; v.gi = it->second.gi; v.gc = it->second.gc; return; }
    analyse_uncached(v);
    if (cache.size() >= (1u << 18)) cache.clear();
    cache.emplace(v.s, v);
}

struct Request {
    std::vector<std::pair<int, Value>> h;                        // in to_request order
    std::vector<std::string> params;
    const Value* get(int f) const {
        for (auto& [k, v] : h) if (k == f) return &v;
        return nullptr;
    }
};

// the 11 SR-BH header columns (run_sse_srbh.HEADER_COLS order) + method, request line, body
struct RawRow {
    std::string hdr[11], method, request, body;
};

inline Request to_request(const RawRow& r) {
    Request q;
    Url p = urlparse(r.request);
    auto add = [&](int f, std::string s) {
        Value v;
        v.s = std::move(s);
        analyse(v);
        q.h.emplace_back(f, std::move(v));
    };
    for (int i = 0; i < 11; i++) if (!r.hdr[i].empty()) add(i, r.hdr[i]);
    parse_qs_keys(p.query, q.params);
    if (!r.body.empty() && r.hdr[CTYPE].find("application/x-www-form-urlencoded") != std::string::npos)
        parse_qs_keys(r.body, q.params);
    add(METHOD, r.method);
    add(PATH, p.path.empty() ? "/" : p.path);
    if (!p.query.empty()) add(QUERY, unquote(p.query, true));
    if (!r.body.empty()) add(BODY, unquote(r.body, true));
    return q;
}

// ------------------------------------------------------------------------------------------------ statistical gate

struct Baseline {                                                // anomaly_model.Baseline
    long N = 0;
    std::vector<int> order;                                      // header names in first-seen order (dict order)
    long names[NFIELD] = {};
    bool seen[NFIELD] = {};
    std::unordered_map<std::string, long> params;
    struct Acc { long n = 0; __int128 s1 = 0, s2 = 0; double s3 = 0, s4 = 0; } acc[NFIELD];   // int sums exact
    std::pair<double, double> len[NFIELD], ent[NFIELD], glen, gent;
    bool has_stat[NFIELD] = {};
    double cent[NFIELD] = {}, cnorm = 1;

    void fit(const std::vector<Request>& reqs, size_t n) {
        N = long(n);
        std::vector<double> lens[NFIELD], ents[NFIELD], allL, allE;
        for (size_t i = 0; i < n; i++) {
            for (auto& [k, v] : reqs[i].h) {
                if (!seen[k]) { seen[k] = true; order.push_back(k); }
                names[k]++;
                lens[k].push_back(v.len);
                ents[k].push_back(v.ent);
            }
            for (auto& p : reqs[i].params) params[p]++;
        }
        for (int k : order) {                                    // allL / allE: lens dict order = first seen
            allL.insert(allL.end(), lens[k].begin(), lens[k].end());
            allE.insert(allE.end(), ents[k].begin(), ents[k].end());
        }
        for (int k : order) {
            Acc& a = acc[k];
            a.n = long(lens[k].size());
            NSum e1, e2;
            for (double x : lens[k]) { a.s1 += int64_t(x); a.s2 += int64_t(x) * int64_t(x); }
            for (double x : ents[k]) { e1.add(x); e2.add(x * x); }
            a.s3 = e1.get(); a.s4 = e2.get();
            len[k] = {exact_mean(lens[k]), std::max(exact_pstdev(lens[k]), 2.0)};
            ent[k] = {exact_mean(ents[k]), std::max(exact_pstdev(ents[k]), 0.3)};
            has_stat[k] = true;
        }
        if (allL.empty()) allL.push_back(0);
        if (allE.empty()) allE.push_back(0);
        double sl = exact_pstdev(allL), se = exact_pstdev(allE);
        glen = {exact_mean(allL), sl != 0 ? sl : 1};
        gent = {exact_mean(allE), se != 0 ? se : 1};
        recentre();
    }

    void recentre() {
        NSum s;
        for (int k : order) { cent[k] = double(names[k]) / double(N); s.add(cent[k] * cent[k]); }
        double c = std::sqrt(s.get());
        cnorm = c != 0 ? c : 1;
    }

    void update(const Request& r) {
        N++;
        for (auto& p : r.params) params[p]++;
        for (auto& [k, v] : r.h) {
            if (!seen[k]) { seen[k] = true; order.push_back(k); }
            names[k]++;
            Acc& a = acc[k];
            double e = v.ent;
            a.n += 1; a.s1 += v.len; a.s2 += int64_t(v.len) * v.len; a.s3 += e; a.s4 += e * e;
            double ml = double(__float128(a.s1) / a.n), me = a.s3 / double(a.n);   // int / int, correctly rounded
            double vl = double(__float128(a.s2) / a.n) - ml * ml, ve = a.s4 / double(a.n) - me * me;
            len[k] = {ml, std::max(std::sqrt(std::max(vl, 0.0)), 2.0)};
            ent[k] = {me, std::max(std::sqrt(std::max(ve, 0.0)), 0.3)};
            has_stat[k] = true;
        }
        recentre();
    }

    double surprisal(long count) const { return -std::log2((double(count) + 0.1) / (double(N) + 0.2)); }
    long pcount(const std::string& p) const { auto it = params.find(p); return it == params.end() ? 0 : it->second; }
};

struct Gate {                                                    // anomaly_model.Gate
    double fpr = 0.01, lr = 0.05, w_min = 0.5, w_max = 3.0;
    double w[NFIELD];
    Baseline base;
    double mu[4], sd[4], thr = 0;
    Gate() { std::fill(w, w + NFIELD, 1.0); }

    std::array<double, 4> raw(const Request& r, bool weighted) const {
        const Baseline& b = base;
        double s1 = 0.0;
        bool first = true;
        auto mx = [&](double x) { if (first || x > s1) s1 = x; first = false; };
        for (auto& [k, v] : r.h) mx(b.surprisal(b.names[k]));
        for (auto& p : r.params) mx(b.surprisal(b.pcount(p)));
        mx(0.0);
        double z_len = 0.0, z_ent = 0.0;
        NSum dot;
        for (auto& [k, v] : r.h) {
            auto [ml, sl] = b.has_stat[k] ? b.len[k] : b.glen;
            auto [me, se] = b.has_stat[k] ? b.ent[k] : b.gent;
            double wk = weighted ? w[k] : 1.0;
            z_len = std::max(z_len, std::fabs(v.len - ml) / sl * wk);
            z_ent = std::max(z_ent, std::fabs(v.ent - me) / se * wk);
            dot.add(b.seen[k] ? b.cent[k] : 0.0);
        }
        double den = std::sqrt(double(r.h.size())) * b.cnorm;
        double cos = dot.get() / (den != 0 ? den : 1);
        return {s1, z_len, z_ent, 1 - cos};
    }

    double score(const Request& r) const {
        auto x = raw(r, true);
        double m = (x[0] - mu[0]) / sd[0];
        for (int i = 1; i < 4; i++) m = std::max(m, (x[i] - mu[i]) / sd[i]);
        return m;
    }

    void header_z(const Request& r, std::vector<std::pair<int, double>>& out) const {
        const Baseline& b = base;
        for (auto& [k, v] : r.h) {
            auto [ml, sl] = b.has_stat[k] ? b.len[k] : b.glen;
            auto [me, se] = b.has_stat[k] ? b.ent[k] : b.gent;
            double comp[3] = {b.surprisal(b.names[k]), std::fabs(v.len - ml) / sl * w[k], std::fabs(v.ent - me) / se * w[k]};
            double m = (comp[0] - mu[0]) / sd[0];
            for (int i = 1; i < 3; i++) m = std::max(m, (comp[i] - mu[i]) / sd[i]);
            out.emplace_back(k, m);
        }
    }

    void fit(const std::vector<Request>& reqs) {
        size_t half = reqs.size() / 2;
        base.fit(reqs, half);
        std::vector<double> cols[4];
        for (size_t i = 0; i < half; i++) {
            auto x = raw(reqs[i], false);
            for (int c = 0; c < 4; c++) cols[c].push_back(x[c]);
        }
        for (int c = 0; c < 4; c++) {
            mu[c] = exact_mean(cols[c]);
            double s = exact_pstdev(cols[c]);
            sd[c] = s != 0 ? s : 1;
        }
        std::vector<double> ns;
        for (size_t i = half; i < reqs.size(); i++) ns.push_back(score(reqs[i]));
        std::sort(ns.begin(), ns.end());
        thr = ns[std::max(long(double(ns.size()) * (1 - fpr)) - 1, 0L)];
    }

    void learn_false_alarm(const Request& r, const std::vector<int>& fired) {
        base.update(r);
        for (int k : fired) if (r.get(k)) w[k] = std::max(w_min, w[k] * (1 - lr));
    }
};

// ------------------------------------------------------------------------------------------------ semantic store

inline void fwht(float* x) {                                     // quant_space.fwht, same butterflies, float32
#if defined(__AVX512F__)
    // h = 1, 2, 4, 8 inside each 16-lane register: lane l pairs with l ^ h; the lower lane keeps a + c, the upper
    // a - c, exactly the scalar butterfly (same operands, same order), so the result is bit-identical
    for (int h = 1; h < 16; h *= 2) {
        alignas(64) int idx[16];
        __mmask16 hi = 0;
        for (int l = 0; l < 16; l++) { idx[l] = l ^ h; if (l & h) hi |= __mmask16(1u << l); }
        __m512i perm = _mm512_load_si512(idx);
        for (int b = 0; b < DIM; b += 16) {
            __m512 v = _mm512_load_ps(x + b), p = _mm512_permutexvar_ps(perm, v);
            _mm512_store_ps(x + b, _mm512_mask_blend_ps(hi, _mm512_add_ps(v, p), _mm512_sub_ps(p, v)));
        }
    }
    for (int h = 16; h < DIM; h *= 2)
        for (int b = 0; b < DIM; b += 2 * h)
            for (int j = b; j < b + h; j += 16) {
                __m512 a = _mm512_load_ps(x + j), c = _mm512_load_ps(x + j + h);
                _mm512_store_ps(x + j, _mm512_add_ps(a, c));
                _mm512_store_ps(x + j + h, _mm512_sub_ps(a, c));
            }
#else
    for (int h = 1; h < DIM; h *= 2)
        for (int b = 0; b < DIM; b += 2 * h)
            for (int j = b; j < b + h; j++) {
                float a = x[j], c = x[j + h];
                x[j] = a + c;
                x[j + h] = a - c;
            }
#endif
}

inline int hamming2048(const uint64_t* a, const uint64_t* b) {
#if defined(__AVX512VPOPCNTDQ__)
    __m512i s = _mm512_setzero_si512();
    for (int j = 0; j < 32; j += 8)
        s = _mm512_add_epi64(s, _mm512_popcnt_epi64(_mm512_xor_si512(_mm512_loadu_si512(a + j), _mm512_loadu_si512(b + j))));
    return int(_mm512_reduce_add_epi64(s));
#else
    int s = 0;
    for (int j = 0; j < 32; j++) s += __builtin_popcountll(a[j] ^ b[j]);
    return s;
#endif
}

// quant_space.CompressedSpace(binary=True, int8=True): per stored value two 3-gram count bit planes (0/1/2/3+),
// exact counts of 3+ in an exception list, a float16 norm and a 2048-bit SRHT sign code.
struct Space {
    static constexpr int BITS = 2048, WORDS = BITS / 64, PLANE = DIM / 8, K = 64, REFINE = 4;
    static constexpr double GSTEP = 2.0 / 255;
    double lr = 0.05, gain_max = 4.0, rare = 0;
    std::vector<_Float16> idf;
    std::vector<double> idf64;
    std::vector<uint8_t> gq;
    std::vector<float> sign;
    std::vector<int> keep;
    double glut[256];
    int n = 0;
    std::vector<uint64_t> C;                                     // n x WORDS
    std::vector<uint8_t> B;                                      // n x 2 x PLANE (plane 0 = bit 0 of the count)
    std::vector<_Float16> norm;
    std::vector<int64_t> ex_key;                                 // row * DIM + gram, ascending
    std::vector<uint16_t> ex_cnt;
    long ver = 0;
    std::unordered_map<std::string, std::pair<long, double>> dc;

    Space(const std::vector<std::string>& values, uint64_t seed) {
        std::vector<std::string> vals(values);                   // HeaderSpace: sorted(set(values))
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        std::vector<Value> vv(vals.size());
        std::vector<double> df(DIM, 0.0);
        for (size_t i = 0; i < vals.size(); i++) {
            vv[i].s = vals[i];
            analyse(vv[i]);
            for (int g : vv[i].gi) df[g] += 1;
        }
        double nv = double(vals.size());
        idf.resize(DIM); idf64.resize(DIM);
        for (int g = 0; g < DIM; g++) {
            idf[g] = _Float16(std::log((nv + 1) / (df[g] + 1)) + 1);
            idf64[g] = double(idf[g]);
        }
        rare = std::log((nv + 1) / (0.01 * nv + 1)) + 1;
        gq.assign(DIM, 0);
        SplitMix64 rng(seed);                                    // quant_space.srht_params
        sign.resize(DIM);
        for (int j = 0; j < DIM; j++) sign[j] = (rng.next() >> 63) ? 1.0f : -1.0f;
        std::vector<int> perm(DIM);
        for (int j = 0; j < DIM; j++) perm[j] = j;
        for (int i = DIM - 1; i > 0; i--) std::swap(perm[i], perm[rng.next() % uint64_t(i + 1)]);
        keep.assign(perm.begin(), perm.begin() + BITS);
        for (int q = 0; q < 256; q++) glut[q] = std::pow(2.0, q * GSTEP);
        for (auto& v : vv) append(v, false);
    }

    // _w: count x idf (x gain), unit length, norm summed in order
    void weights(const Value& v, bool gain, std::vector<double>& w) const {
        w.resize(v.gi.size());
        double s = 0;
        for (size_t i = 0; i < w.size(); i++) {
            w[i] = double(v.gc[i]) * idf64[v.gi[i]] * (gain ? glut[gq[v.gi[i]]] : 1.0);
            s += w[i] * w[i];
        }
        double nrm = std::sqrt(s);
        if (nrm == 0) nrm = 1;
        for (auto& x : w) x = x / nrm;
    }

    void code(const Value& v, const std::vector<double>& w, uint64_t* out) const {
        alignas(64) float x[DIM];
        std::memset(x, 0, sizeof x);
        for (size_t i = 0; i < w.size(); i++) x[v.gi[i]] = float(w[i]);
        for (int j = 0; j < DIM; j++) x[j] = x[j] * sign[j];
        fwht(x);
#if defined(__AVX512F__)
        auto* o16 = reinterpret_cast<uint16_t*>(out);            // 16 kept coordinates per gather, bit b of word b/64
        for (int b = 0; b < BITS; b += 16) {
            __m512 g = _mm512_i32gather_ps(_mm512_loadu_si512(&keep[b]), x, 4);
            o16[b >> 4] = _mm512_cmp_ps_mask(g, _mm512_setzero_ps(), _CMP_GT_OQ);
        }
#else
        std::memset(out, 0, WORDS * 8);
        for (int b = 0; b < BITS; b++) if (x[keep[b]] > 0) out[b >> 6] |= 1ull << (b & 63);
#endif
    }

    void append(const Value& v, bool gain) {                     // _append_values for one value
        C.resize(size_t(n + 1) * WORDS);
        B.resize(size_t(n + 1) * 2 * PLANE, 0);
        uint8_t* b0 = &B[size_t(n) * 2 * PLANE];
        uint8_t* b1 = b0 + PLANE;
        double s = 0;
        for (size_t i = 0; i < v.gi.size(); i++) {
            int g = v.gi[i], c = v.gc[i], q = std::min(c, 3);
            if (q & 1) b0[g >> 3] |= uint8_t(1u << (g & 7));
            if (q & 2) b1[g >> 3] |= uint8_t(1u << (g & 7));
            if (c >= 3) { ex_key.push_back(int64_t(n) * DIM + g); ex_cnt.push_back(uint16_t(std::min(c, 65535))); }
            double x = double(c) * idf64[g];
            s += x * x;
        }
        norm.push_back(_Float16(std::sqrt(s)));
        std::vector<double> w;
        weights(v, gain, w);
        code(v, w, &C[size_t(n) * WORDS]);
        n++;
    }

    double dist(const Value& v) {
        auto it = dc.find(v.s);
        if (it != dc.end() && it->second.first == ver) return it->second.second;
        double d = dist_uncached(v);
        if (dc.size() > 100000) dc.clear();
        dc[v.s] = {ver, d};
        return d;
    }

    static inline double prof[6] = {};                           // SSE_PROFILE: us in weights, code, hamming, select, rescore
    static inline bool profiling = false;

    double dist_uncached(const Value& v) const {
        using clk = std::chrono::steady_clock;
        auto t0 = profiling ? clk::now() : clk::time_point{};
        auto lap = [&](int i) {
            if (!profiling) return;
            auto t = clk::now();
            prof[i] += std::chrono::duration<double, std::micro>(t - t0).count();
            t0 = t;
        };
        std::vector<double> w;
        weights(v, true, w);
        lap(0);
        alignas(64) uint64_t q[WORDS];
        code(v, w, q);
        lap(1);
        static thread_local std::vector<uint16_t> ham;
        ham.resize(n);
        uint32_t hist[BITS + 1] = {};
        for (int r = 0; r < n; r++) {                            // XOR + popcount over every stored code
            int s = hamming2048(&C[size_t(r) * WORDS], q);
            ham[r] = uint16_t(s);
            hist[s]++;
        }
        lap(2);
        int k = std::min(K, n), kth = 0;
        for (uint32_t acc = 0; kth <= BITS; kth++) { acc += hist[kth]; if (acc >= uint32_t(k)) break; }
        int less = 0;
        for (int h = 0; h < kth; h++) less += hist[h];
        int cand[K], nc = 0, ties = k - less;
        for (int r = 0; r < n && nc < k; r++) {                  // all closer than the k-th, then ties by row
            if (ham[r] < kth) cand[nc++] = r;
            else if (ham[r] == kth && ties > 0) { cand[nc++] = r; ties--; }
        }
        std::sort(cand, cand + nc);
        lap(3);
        size_t m = v.gi.size();
        static thread_local std::vector<double> t;
        static thread_local std::vector<int32_t> tq, cnt;
        static thread_local std::vector<int> byte;
        static thread_local std::vector<uint8_t> bit;
        t.resize(m); tq.resize(m); byte.resize(m); bit.resize(m); cnt.resize(size_t(nc) * m);
        double tmax = 0;
        for (size_t i = 0; i < m; i++) {
            t[i] = w[i] * idf64[v.gi[i]];
            tmax = std::max(tmax, t[i]);
            byte[i] = v.gi[i] >> 3;
            bit[i] = uint8_t(v.gi[i] & 7);
        }
        double scale = 127.0 / tmax;
        for (size_t i = 0; i < m; i++) tq[i] = int32_t(std::nearbyint(t[i] * scale));
        double cos8[K], nrm[K];
        for (int j = 0; j < nc; j++) {                           // u8 counts x s8 query, int32 dot
            const uint8_t* b0 = &B[size_t(cand[j]) * 2 * PLANE];
            const uint8_t* b1 = b0 + PLANE;
            int32_t* cj = &cnt[size_t(j) * m];
            int32_t dot = 0;
            bool three = false;
            for (size_t i = 0; i < m; i++) {
                int c = ((b0[byte[i]] >> bit[i]) & 1) | (((b1[byte[i]] >> bit[i]) & 1) << 1);
                cj[i] = c;
                three |= c == 3;
            }
            if (three) {                                         // exact counts of 3+: this row's exception keys
                int64_t lo = int64_t(cand[j]) * DIM;
                auto it = std::lower_bound(ex_key.begin(), ex_key.end(), lo);
                for (size_t i = 0; i < m; i++)
                    if (cj[i] == 3) {
                        while (it != ex_key.end() && *it < lo + v.gi[i]) ++it;
                        cj[i] = ex_cnt[(it == ex_key.end() ? ex_key.end() - 1 : it) - ex_key.begin()];
                    }
            }
            for (size_t i = 0; i < m; i++) dot += std::min(cj[i], 255) * tq[i];
            nrm[j] = std::max(double(norm[cand[j]]), 1e-6);
            cos8[j] = double(dot) / scale / nrm[j];
        }
        int ord[K];
        for (int j = 0; j < nc; j++) ord[j] = j;
        std::sort(ord, ord + nc, [&](int a, int b) { return cos8[a] != cos8[b] ? cos8[a] > cos8[b] : cand[a] < cand[b]; });
        double best = -INFINITY;
        for (int r = 0; r < std::min(REFINE, nc); r++) {         // exact refine of the int8 top REFINE
            int j = ord[r];
            double s = 0;
            for (size_t i = 0; i < m; i++) s += double(cnt[size_t(j) * m + i]) * t[i];
            best = std::max(best, s / nrm[j]);
        }
        lap(4);
        prof[5] += profiling;
        return 1 - best;
    }

    void add(const Value& v) { ver++; append(v, true); }

    void sharpen(const Value& v, double lr_) {
        ver++;
        int up = int(std::nearbyint(std::log2(1 + lr_) / GSTEP)), cap = int(std::nearbyint(std::log2(gain_max) / GSTEP));
        for (int g : v.gi) if (double(idf[g]) >= rare) gq[g] = uint8_t(std::min(cap, gq[g] + up));
    }

    void soften(const Value& v) {
        ver++;
        int down = int(std::nearbyint(-std::log2(1 - lr) / GSTEP));
        for (int g : v.gi) gq[g] = uint8_t(std::max(0, gq[g] - down));
    }

    size_t nbytes() const {
        return size_t(n) * 2 * PLANE + size_t(n) * 2 + ex_key.size() * 10 + size_t(n) * WORDS * 8 + DIM * 2 + DIM;
    }
};

struct Semantic {
    double fpr = 0.01;
    int min_values = 20;
    std::vector<std::unique_ptr<Space>> space;                   // by field, null where none
    std::pair<double, double> stat[NFIELD];
    bool has_stat[NFIELD] = {};
    double thr = 0;
    Semantic() : space(NFIELD) {}

    // Semantic.fit + quant_space.compress: fields with >= min_values values in the first half, seeds in name order
    void fit(const std::vector<Request>& reqs) {
        size_t half = reqs.size() / 2;
        std::vector<std::string> byname[NFIELD];
        for (size_t i = 0; i < half; i++) for (auto& [k, v] : reqs[i].h) byname[k].push_back(v.s);
        std::vector<int> fields;
        for (int k = 0; k < NFIELD; k++) if (int(byname[k].size()) >= min_values) fields.push_back(k);
        std::sort(fields.begin(), fields.end(), [](int a, int b) { return std::string(FIELD_NAMES[a]) < FIELD_NAMES[b]; });
        for (size_t i = 0; i < fields.size(); i++) space[fields[i]] = std::make_unique<Space>(byname[fields[i]], i);
        std::vector<double> d[NFIELD];
        for (size_t i = half; i < reqs.size(); i++)
            for (auto& [k, v] : reqs[i].h) if (space[k]) d[k].push_back(space[k]->dist(v));
        for (int k = 0; k < NFIELD; k++)
            if (!d[k].empty()) { stat[k] = {exact_mean(d[k]), std::max(exact_pstdev(d[k]), 0.02)}; has_stat[k] = true; }
        std::vector<double> ns;
        std::vector<std::pair<int, double>> z;
        for (size_t i = half; i < reqs.size(); i++) {
            z.clear();
            header_z(reqs[i], z);
            double m = 0.0;                                      // Semantic.score: max(z values + [0.0])
            for (auto& [k, x] : z) m = std::max(m, x);
            ns.push_back(m);
        }
        std::sort(ns.begin(), ns.end());
        thr = ns[std::max(long(double(ns.size()) * (1 - fpr)) - 1, 0L)];
    }

    void header_z(const Request& r, std::vector<std::pair<int, double>>& out) {
        for (auto& [k, v] : r.h)
            if (has_stat[k]) out.emplace_back(k, (space[k]->dist(v) - stat[k].first) / stat[k].second);
    }

    size_t nbytes() const {
        size_t s = 0;
        for (auto& p : space) if (p) s += p->nbytes();
        return s;
    }
};

// ------------------------------------------------------------------------------------------------ token thresholds

inline std::vector<std::string> route_tokens(const std::string& rt) {
    std::string low = lower_utf8(rt);
    std::vector<std::string> segs;
    size_t s = 0;
    while (s <= low.size()) {
        size_t e = low.find('/', s);
        if (e == std::string::npos) e = low.size();
        if (e > s) {
            std::string x;
            for (size_t i = s; i < e;) {
                if (low[i] >= '0' && low[i] <= '9') { x += '0'; while (i < e && low[i] >= '0' && low[i] <= '9') i++; }
                else x += low[i++];
            }
            segs.push_back(x);
        }
        s = e + 1;
    }
    std::vector<std::string> t = {"*", "r:" + rt, "n:" + std::to_string(std::min<size_t>(segs.size(), 6))};
    for (size_t i = 0; i < std::min<size_t>(segs.size(), 8); i++) t.push_back("s:" + segs[i]);
    if (!segs.empty()) {
        t.push_back("p1:" + segs[0]);
        if (segs.size() > 1) t.push_back("p2:" + segs[0] + "/" + segs[1]);
        size_t dot = segs.back().rfind('.');
        if (dot != std::string::npos) t.push_back("e:" + segs.back().substr(dot + 1));
    }
    std::vector<std::string> u;                                  // dict.fromkeys: dedupe, keep order
    for (auto& x : t) if (std::find(u.begin(), u.end(), x) == u.end()) u.push_back(x);
    return u;
}

// replay_hybrid2.TokenThresholds(int8=True, packed=True, portable=True)
struct TokenThresholds {
    static constexpr int BITSZ = 20;
    static constexpr uint32_t MASK = (1u << BITSZ) - 1;
    double eta = 0.5, margin = 0.25, cap = 4.0, step = 4.0 / 127;
    std::vector<int8_t> d;
    std::vector<uint8_t> st;                                     // bits 0-5 flips, 6 moved up, 7 moved down
    SplitMix64 rng{0};
    std::string memo_rt;                                         // one-route memo: tokens and their table slots
    bool memo_ok = false;
    std::vector<std::string> memo_toks;
    std::vector<uint32_t> memo_ids[2][NFIELD + 1];
    bool memo_have[2][NFIELD + 1] = {};
    uint32_t prefix[2][NFIELD + 1];                              // crc register after "layer|field|"
    // exact layer (exact = true): one int8 correction per (layer, field, exact route) in a fixed hashed table, set in
    // full to just past the misjudged score after the token move (the published hybrid's exact overrides, fixed size)
    static constexpr int EBITS = 22;
    static constexpr uint32_t EMASK = (1u << EBITS) - 1;
    bool exact = false;
    double estep = 8.0 / 127;
    std::vector<int8_t> E;
    uint32_t memo_eid[2][NFIELD + 1];
    bool memo_ehave[2][NFIELD + 1] = {};
    TokenThresholds() : d(1u << BITSZ, 0), st(1u << BITSZ, 0) {
        for (int l = 0; l < 2; l++)
            for (int f = 0; f <= NFIELD; f++) {
                std::string p = std::string(l ? "sem" : "gate") + "|" + FIELD_NAMES[f] + "|";
                prefix[l][f] = crc32_update(0xFFFFFFFFu, reinterpret_cast<const uint8_t*>(p.data()), p.size());
            }
    }

    const std::vector<uint32_t>& ids(const std::string& rt, const char* layer, int field) {
        if (!memo_ok || rt != memo_rt) {
            memo_rt = rt; memo_toks = route_tokens(rt); memo_ok = true;
            for (auto& row : memo_have) std::fill(row, row + NFIELD + 1, false);
            for (auto& row : memo_ehave) std::fill(row, row + NFIELD + 1, false);
        }
        int l = layer[0] == 's';
        auto& out = memo_ids[l][field];
        if (!memo_have[l][field]) {
            out.clear();
            for (auto& t : memo_toks)
                out.push_back((crc32_update(prefix[l][field], reinterpret_cast<const uint8_t*>(t.data()), t.size()) ^ 0xFFFFFFFFu) & MASK);
            memo_have[l][field] = true;
        }
        return out;
    }

    uint32_t eid(const std::string& rt, const char* layer, int field) {   // crc32("layer|field|x:" + route)
        ids(rt, layer, field);                                   // refreshes the one-route memo
        int l = layer[0] == 's';
        if (!memo_ehave[l][field]) {
            std::string k = "x:" + rt;
            memo_eid[l][field] = (crc32_update(prefix[l][field], reinterpret_cast<const uint8_t*>(k.data()), k.size()) ^ 0xFFFFFFFFu) & EMASK;
            memo_ehave[l][field] = true;
        }
        return memo_eid[l][field];
    }

    double thr(const std::string& rt, const char* layer, int field, double base) {
        const auto& i = ids(rt, layer, field);
        int64_t s = 0;
        for (uint32_t x : i) s += d[x];
        double t = base + double(s) * step / double(i.size());
        if (exact) t = t + double(E[eid(rt, layer, field)]) * estep;
        return t;
    }

    void pin(const std::string& rt, const char* layer, int field, double target, double base, bool down) {
        double t2 = thr(rt, layer, field, base);
        if (down ? !(target < t2) : !(target > t2)) return;
        uint32_t e = eid(rt, layer, field);
        double x = double(E[e]) + (target - t2) / estep;
        x = down ? std::floor(x) : std::ceil(x);
        E[e] = int8_t(std::min(std::max(x, -127.0), 127.0));
    }

    void move(const std::string& rt, const char* layer, int field, double delta) {
        const std::vector<uint32_t> i = ids(rt, layer, field);
        bool upw = delta > 0;
        size_t n = i.size();
        std::vector<int> flips(n);
        for (size_t j = 0; j < n; j++) {
            int s = st[i[j]];
            flips[j] = std::min((s & 63) + ((s & (upw ? 128 : 64)) > 0 ? 1 : 0), 63);
        }
        for (size_t j = 0; j < n; j++) st[i[j]] = uint8_t(flips[j] | (upw ? 64 : 128));
        std::vector<double> x(n);
        for (size_t j = 0; j < n; j++) {
            double e = eta / (1.0 + std::min(flips[j], 9));
            double y = double(d[i[j]]) * step + e * delta;
            y = std::min(std::max(y, -cap), cap);
            x[j] = y / step;
        }
        for (size_t j = 0; j < n; j++) {
            double f = std::floor(x[j]);
            double r = rng.uniform();
            double y = f + (r < x[j] - f ? 1.0 : 0.0);
            x[j] = std::min(std::max(y, -127.0), 127.0);
        }
        for (size_t j = 0; j < n; j++) d[i[j]] = int8_t(x[j]);
    }

    void tighten(const std::string& rt, const char* layer, int field, double z, double base) {
        double t = thr(rt, layer, field, base);
        if (z - margin < t) {
            move(rt, layer, field, z - margin - t);
            if (exact) pin(rt, layer, field, z - margin, base, true);
        }
    }
    void loosen(const std::string& rt, const char* layer, int field, double z, double base) {
        double t = thr(rt, layer, field, base);
        if (z + margin > t) {
            move(rt, layer, field, z + margin - t);
            if (exact) pin(rt, layer, field, z + margin, base, false);
        }
    }
    long exact_set() const { return std::count_if(E.begin(), E.end(), [](int8_t x) { return x != 0; }); }
    long entries_set() const { return std::count_if(d.begin(), d.end(), [](int8_t x) { return x != 0; }); }
    long entries_flipped() const { return std::count_if(st.begin(), st.end(), [](uint8_t x) { return (x & 63) > 0; }); }
    size_t bytes() const { return d.size() + st.size() + E.size(); }
};

// ------------------------------------------------------------------------------------------------ the hybrid

struct Hybrid {                                                  // replay_hybrid2.Hybrid2(balanced=True, tokens=True)
    Gate gate;
    Semantic sem;
    TokenThresholds tok;
    double lr_miss = 0.15;

    struct Feat { std::vector<std::pair<int, double>> zg, zs; };

    void features(const Request& r, Feat& f) {
        f.zg.clear(); f.zs.clear();
        gate.header_z(r, f.zg);
        f.zg.emplace_back(REQ, gate.score(r));
        sem.header_z(r, f.zs);
    }

    void over(const std::string& rt, const char* layer, const std::vector<std::pair<int, double>>& z, double base,
              std::vector<int>& out) {
        out.clear();
        for (auto& [k, x] : z) if (x > tok.thr(rt, layer, k, base)) out.push_back(k);
    }

    // one request: verdict, then (learn) correct a wrong verdict. Returns the verdict; *wrong set if corrected.
    bool step(const std::string& rt, const Request& r, bool attack, bool learn, bool check, int* flipped) {
        Feat f;
        features(r, f);
        std::vector<int> og, os;
        over(rt, "gate", f.zg, gate.thr, og);
        over(rt, "sem", f.zs, sem.thr, os);
        bool flagged = !og.empty() || !os.empty();
        if (flagged == attack || !learn) return flagged;
        if (attack) {
            int blamed = -1;
            double bz = 0;
            for (auto& [k, z] : f.zg) if (k != REQ && (blamed < 0 || z > bz)) { blamed = k; bz = z; }
            std::vector<int> up;
            for (auto& [k, z] : f.zg) if (k != REQ && z >= 0.5 * gate.thr) up.push_back(k);
            if (up.empty() && blamed >= 0) up.push_back(blamed);
            for (int k : up) gate.w[k] = std::min(gate.w_max, gate.w[k] * (1 + lr_miss));
            std::vector<int> sup;
            for (auto& [k, z] : f.zs) if (z >= 0.5 * sem.thr) sup.push_back(k);
            if (sup.empty() && !f.zs.empty()) {
                int a = f.zs[0].first;
                double az = f.zs[0].second;
                for (auto& [k, z] : f.zs) if (z > az) { a = k; az = z; }
                sup.push_back(a);
            }
            for (int k : sup) sem.space[k]->sharpen(*r.get(k), lr_miss);
            if (blamed >= 0) {
                tok.tighten(rt, "gate", blamed, bz, gate.thr);
                for (auto& [k, z] : f.zs) if (k == blamed) { tok.tighten(rt, "sem", blamed, z, sem.thr); break; }
            }
        } else {
            std::vector<int> fired;
            for (int k : og) if (k != REQ) fired.push_back(k);
            gate.learn_false_alarm(r, fired);
            for (int k : os) {
                const Value* v = r.get(k);
                if (sem.space[k] && v) { sem.space[k]->soften(*v); sem.space[k]->add(*v); }
            }
            auto zof = [](const std::vector<std::pair<int, double>>& z, int k) {
                for (auto& [kk, x] : z) if (kk == k) return x;
                return 0.0;
            };
            for (int k : og) tok.loosen(rt, "gate", k, zof(f.zg, k), gate.thr);
            for (int k : os) tok.loosen(rt, "sem", k, zof(f.zs, k), sem.thr);
        }
        if (check && flipped) {
            Feat f2;
            features(r, f2);
            over(rt, "gate", f2.zg, gate.thr, og);
            over(rt, "sem", f2.zs, sem.thr, os);
            *flipped = (!og.empty() || !os.empty()) == attack;
        }
        return flagged;
    }
};

}  // namespace sse
