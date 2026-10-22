// replay_srbh.cpp: SR-BH 2020 lap 1 of the prototype hybrid, fully binary semantic store + int8 rescore, in C++.
//
// Same protocol as harness/replay_hybrid2.py with HYB_QUANT=3 (variant "full"): rows sorted by time, 10,000 normal
// 17 Jul requests drawn with Python's random.Random(7).shuffle fit the gates, the rest of 17 Jul precalibrates,
// 18-19 Jul (traffic never seen) is lap 1, learning from every wrong verdict (teacher = dataset label).
// Then a frozen pass over the same 18-19 Jul stream (no learning, caches cleared) times inference alone.
//
// usage: replay_srbh <data_capec_multilabel.csv> <out.json> [trace.txt] [laps]
//        replay_srbh --prepared <stream.csv> <out.json> [trace.txt] [laps]   (splits given in the file)
//        replay_srbh --features <data_capec_multilabel.csv> <out.txt>   (per-row feature digests, parity checks)
#include <chrono>
#include <clocale>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>

#include "sse.hpp"

using namespace sse;

namespace {

struct Row {
    int split = -1;                                              // --prepared: 0 fit, 1 precal, 2 test
    int64_t ts;
    bool normal;
    std::vector<int> labels;                                     // attack label ids
    RawRow raw;
};

// csv module, excel dialect, strict=False: quoted fields may span lines, "" is a quote, text after a closing quote
// is kept, a quote inside an unquoted field is literal
struct CsvReader {
    const char* p;
    const char* end;
    bool next(std::vector<std::string>& out) {
        out.clear();
        if (p >= end) return false;
        std::string f;
        enum { START, UNQ, QUO, QQ } st = START;
        bool any = false;
        while (p < end) {
            char c = *p++;
            if (st == QUO) {
                if (c == '"') st = QQ; else f += c;
                continue;
            }
            if (c == '\r' || c == '\n') {
                if (c == '\r' && p < end && *p == '\n') p++;
                if (st == START && !any && f.empty() && out.empty()) return true;   // blank line: empty record
                out.push_back(std::move(f));
                return true;
            }
            any = true;
            if (st == QQ) {
                if (c == '"') { f += '"'; st = QUO; }
                else if (c == ',') { out.push_back(std::move(f)); f.clear(); st = START; }
                else { f += c; st = UNQ; }
            } else if (st == START) {
                if (c == '"') st = QUO;
                else if (c == ',') { out.push_back(std::move(f)); f.clear(); }
                else { f += c; st = UNQ; }
            } else {
                if (c == ',') { out.push_back(std::move(f)); f.clear(); st = START; }
                else f += c;
            }
        }
        if (any || !f.empty() || !out.empty()) out.push_back(std::move(f));
        return true;
    }
};

int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = unsigned(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + int64_t(doe) - 719468;
}

// "%d/%b/%Y:%H:%M:%S %z" -> epoch seconds; false when strptime would raise
bool parse_ts(const std::string& s, int64_t& out) {
    static const char* mon[] = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"};
    size_t i = 0;
    auto num = [&](int lo, int hi, int& v) {
        int k = 0;
        v = 0;
        while (i < s.size() && k < hi && std::isdigit(uint8_t(s[i]))) { v = v * 10 + (s[i++] - '0'); k++; }
        return k >= lo;
    };
    auto lit = [&](char c) { return i < s.size() && s[i++] == c; };
    int d, y, H, M, S, m = -1;
    if (!num(1, 2, d) || !lit('/')) return false;
    if (i + 3 > s.size()) return false;
    for (int k = 0; k < 12; k++)
        if (std::tolower(uint8_t(s[i])) == mon[k][0] && std::tolower(uint8_t(s[i + 1])) == mon[k][1] &&
            std::tolower(uint8_t(s[i + 2])) == mon[k][2]) m = k + 1;
    if (m < 0) return false;
    i += 3;
    if (!lit('/') || !num(4, 4, y) || !lit(':') || !num(1, 2, H) || !lit(':') || !num(1, 2, M) || !lit(':') ||
        !num(1, 2, S) || !lit(' '))
        return false;
    if (i >= s.size() || (s[i] != '+' && s[i] != '-')) return false;
    int sg = s[i++] == '-' ? -1 : 1, zh, zm;
    if (!num(2, 2, zh)) return false;
    if (i < s.size() && s[i] == ':') i++;
    if (!num(2, 2, zm) || i != s.size() || d < 1 || d > 31 || H > 23 || M > 59 || S > 61 || zm > 59) return false;
    out = days_from_civil(y, m, d) * 86400 + H * 3600 + M * 60 + S - sg * (zh * 3600 + zm * 60);
    return true;
}

struct PyRandom {                                                // CPython random.Random(int seed): MT19937
    uint32_t mt[624];
    int mti = 624;
    explicit PyRandom(uint32_t seed) {
        mt[0] = 19650218u;
        for (int i = 1; i < 624; i++) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + uint32_t(i);
        uint32_t key[1] = {seed};
        int i = 1, j = 0;
        for (int k = 624; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525u)) + key[j] + uint32_t(j);
            i++; j++;
            if (i >= 624) { mt[0] = mt[623]; i = 1; }
            if (j >= 1) j = 0;
        }
        for (int k = 623; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941u)) - uint32_t(i);
            i++;
            if (i >= 624) { mt[0] = mt[623]; i = 1; }
        }
        mt[0] = 0x80000000u;
    }
    uint32_t next() {
        if (mti >= 624) {
            for (int k = 0; k < 624; k++) {
                uint32_t y = (mt[k] & 0x80000000u) | (mt[(k + 1) % 624] & 0x7fffffffu);
                mt[k] = mt[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0);
            }
            mti = 0;
        }
        uint32_t y = mt[mti++];
        y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680u; y ^= (y << 15) & 0xefc60000u; y ^= y >> 18;
        return y;
    }
    uint32_t randbelow(uint32_t n) {                             // _randbelow_with_getrandbits, n < 2^32
        int k = 32 - __builtin_clz(n);
        uint32_t r = next() >> (32 - k);
        while (r >= n) r = next() >> (32 - k);
        return r;
    }
    template <class T> void shuffle(std::vector<T>& x) {
        for (size_t i = x.size() - 1; i >= 1; i--) std::swap(x[i], x[randbelow(uint32_t(i + 1))]);
    }
};

std::vector<std::string> g_label_names;

std::vector<Row> load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CsvReader rd{buf.data(), buf.data() + buf.size()};
    std::vector<std::string> hdr, rec;
    rd.next(hdr);
    for (auto& h : hdr) h = sanitize_utf8(h);
    auto col = [&](const std::string& name) {
        for (size_t i = 0; i < hdr.size(); i++) if (hdr[i] == name) return int(i);
        std::cerr << "missing column " << name << "\n";
        std::exit(1);
    };
    static const char* HC[] = {"request_user_agent", "request_referer", "request_host", "request_origin",
                               "request_cookie", "request_content_type", "request_accept",
                               "request_accept_language", "request_accept_encoding", "request_do_not_track",
                               "request_connection"};
    int ch[11];
    for (int i = 0; i < 11; i++) ch[i] = col(HC[i]);
    int cts = col("timestamp"), cm = col("request_http_method"), cr = col("request_http_request"),
        cb = col("request_body"), cn = col("000 - Normal");
    std::vector<int> lab;                                        // label columns other than Normal
    for (size_t i = 0; i < hdr.size(); i++) {
        auto& h = hdr[i];
        size_t k = 0;
        while (k < h.size() && std::isdigit(uint8_t(h[k]))) k++;
        if (k > 0 && h.compare(k, 3, " - ") == 0 && int(i) != cn) { lab.push_back(int(i)); g_label_names.push_back(h); }
    }
    int unclassified = int(g_label_names.size());
    g_label_names.push_back("anomalous, no CAPEC class");
    std::vector<Row> rows;
    rows.reserve(1000000);
    long short_rows = 0;
    while (rd.next(rec)) {
        if (rec.empty()) continue;                               // DictReader skips blank rows
        if (rec.size() < hdr.size()) { short_rows++; rec.resize(hdr.size()); }
        Row r;
        if (!parse_ts(sanitize_utf8(rec[cts]), r.ts)) continue;
        r.normal = rec[cn] == "1";
        if (!r.normal) {
            for (size_t j = 0; j < lab.size(); j++) if (rec[lab[j]] == "1") r.labels.push_back(int(j));
            if (r.labels.empty()) r.labels.push_back(unclassified);
        }
        for (int i = 0; i < 11; i++) r.raw.hdr[i] = sanitize_utf8(rec[ch[i]]);
        r.raw.method = sanitize_utf8(rec[cm]);
        r.raw.request = sanitize_utf8(rec[cr]);
        r.raw.body = sanitize_utf8(rec[cb]);
        rows.push_back(std::move(r));
    }
    if (short_rows) std::cerr << "warning: " << short_rows << " short rows padded\n";
    std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.ts < b.ts; });
    return rows;
}

// --prepared: a stream already split and ordered (bench/sse_prepare.py): columns split (fit / precal / test),
// label (0 normal, 1 attack), class, request_http_method, request_http_request, request_body and the 11 header
// columns. Rows keep file order.
std::vector<Row> load_prepared(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CsvReader rd{buf.data(), buf.data() + buf.size()};
    std::vector<std::string> hdr, rec;
    rd.next(hdr);
    auto col = [&](const std::string& name) {
        for (size_t i = 0; i < hdr.size(); i++) if (hdr[i] == name) return int(i);
        std::cerr << "missing column " << name << "\n";
        std::exit(1);
    };
    static const char* HC[] = {"request_user_agent", "request_referer", "request_host", "request_origin",
                               "request_cookie", "request_content_type", "request_accept",
                               "request_accept_language", "request_accept_encoding", "request_do_not_track",
                               "request_connection"};
    int ch[11];
    for (int i = 0; i < 11; i++) ch[i] = col(HC[i]);
    int cs = col("split"), cl = col("label"), cc = col("class"), cm = col("request_http_method"),
        cr = col("request_http_request"), cb = col("request_body");
    std::map<std::string, int> cls;
    std::vector<Row> rows;
    while (rd.next(rec)) {
        if (rec.empty()) continue;
        rec.resize(hdr.size());
        Row r;
        r.split = rec[cs] == "fit" ? 0 : rec[cs] == "precal" ? 1 : 2;
        r.ts = int64_t(rows.size());
        r.normal = rec[cl] != "1";
        if (!r.normal) {
            std::string c = rec[cc].empty() ? "attack" : rec[cc];
            auto it = cls.find(c);
            if (it == cls.end()) { it = cls.emplace(c, int(g_label_names.size())).first; g_label_names.push_back(c); }
            r.labels.push_back(it->second);
        }
        for (int i = 0; i < 11; i++) r.raw.hdr[i] = sanitize_utf8(rec[ch[i]]);
        r.raw.method = sanitize_utf8(rec[cm]);
        r.raw.request = sanitize_utf8(rec[cr]);
        r.raw.body = sanitize_utf8(rec[cb]);
        rows.push_back(std::move(r));
    }
    return rows;
}

const int64_t DAY2 = days_from_civil(2020, 7, 18) * 86400 - 3600;          // 18/Jul/2020:00:00:00 +0100
const int64_t ATTACK_END = days_from_civil(2020, 7, 20) * 86400 - 3600;    // 20/Jul/2020:00:00:00 +0100

std::string route_of(const RawRow& r) {
    Url u = urlparse(r.request.empty() ? std::string("/") : r.request);
    return u.path.empty() ? "/" : u.path;
}

struct LapResult {
    long tp = 0, fn = 0, fp = 0, tn = 0, corrected = 0, checked = 0, flipped = 0;
    std::map<int, std::pair<long, long>> per_class;              // label -> (hit, miss)
    double seconds = 0;
    double tpr() const { return double(tp) / std::max(1L, tp + fn); }
    double fpr() const { return double(fp) / std::max(1L, fp + tn); }
    double precision() const { return double(tp) / std::max(1L, tp + fp); }
    long wrong() const { return fn + fp; }
};

LapResult run(Hybrid& det, const std::vector<const Row*>& stream, const char* label, std::string* trace) {
    LapResult L;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < stream.size(); i++) {
        const Row& row = *stream[i];
        Request q = to_request(row.raw);
        std::string rt = route_of(row.raw);
        bool attack = !row.normal, check = L.corrected % 100 == 0;
        int flipped = -1;
        bool flagged = det.step(rt, q, attack, true, check, &flipped);
        if (flagged != attack) {
            L.corrected++;
            if (check) { L.checked++; L.flipped += flipped == 1; }
        }
        if (trace) *trace += flagged ? '1' : '0';
        (attack ? (flagged ? L.tp : L.fn) : (flagged ? L.fp : L.tn))++;
        if (attack) for (int c : row.labels) (flagged ? L.per_class[c].first : L.per_class[c].second)++;
        if (i % 50000 == 0 && i) {
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "    %s %zu/%zu tpr=%.3f fpr=%.3f %.0fs\n", label, i, stream.size(), L.tpr(), L.fpr(), s);
        }
    }
    L.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return L;
}

std::string jnum(double x) {
    char b[64];
    std::snprintf(b, sizeof b, "%.10g", x);
    return b;
}

std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (uint8_t(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o + "\"";
}

std::string lap_json(const LapResult& L, size_t n) {
    std::ostringstream o;
    o << "{\"tpr\": " << jnum(L.tpr()) << ", \"fpr\": " << jnum(L.fpr()) << ", \"precision\": " << jnum(L.precision())
      << ", \"wrong\": " << L.wrong() << ", \"tp\": " << L.tp << ", \"fn\": " << L.fn << ", \"fp\": " << L.fp
      << ", \"tn\": " << L.tn << ", \"corrected\": " << L.corrected << ", \"flipped_rate\": "
      << jnum(double(L.flipped) / std::max(1L, L.checked)) << ", \"seconds\": " << jnum(L.seconds)
      << ", \"ms_per_request\": " << jnum(n ? L.seconds / n * 1e3 : 0.0) << ", \"per_class_tpr\": {";
    std::vector<std::pair<std::string, double>> pc;
    for (auto& [c, hm] : L.per_class) pc.push_back({g_label_names[c], double(hm.first) / (hm.first + hm.second)});
    std::sort(pc.begin(), pc.end());
    for (size_t i = 0; i < pc.size(); i++) o << (i ? ", " : "") << jstr(pc[i].first) << ": " << jnum(pc[i].second);
    o << "}}";
    return o.str();
}

int features_dump(const std::string& csv, const std::string& out) {
    auto rows = load(csv);
    std::ofstream f(out);
    for (auto& r : rows) {
        Request q = to_request(r.raw);
        std::string s;
        char b[64];
        for (auto& [k, v] : q.h) {
            std::snprintf(b, sizeof b, "\x1e%d\x1e%.17g\x1e", v.len, v.ent);
            s += FIELD_NAMES[k]; s += "\x1e"; s += v.s; s += b;
            for (size_t i = 0; i < v.gi.size(); i++) s += std::to_string(v.gi[i]) + ":" + std::to_string(v.gc[i]) + ",";
            s += "\x1d";
        }
        s += "\x1c";
        for (auto& p : q.params) { s += p; s += "\x1d"; }
        s += "\x1c" + route_of(r.raw);
        f << r.ts << " " << crc32(s) << "\n";
    }
    std::cerr << rows.size() << " rows\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setlocale(LC_CTYPE, "C.UTF-8");
    if (argc >= 4 && std::string(argv[1]) == "--features") return features_dump(argv[2], argv[3]);
    bool prepared = argc > 1 && std::string(argv[1]) == "--prepared";
    if (prepared) { argv++; argc--; }
    if (argc < 3) {
        std::cerr << "usage: replay_srbh <data_capec_multilabel.csv> <out.json> [trace.txt] [laps]\n";
        return 2;
    }
    std::string csv = argv[1], outp = argv[2], tracep = argc > 3 ? argv[3] : "";
    int laps = argc > 4 ? std::atoi(argv[4]) : 1;
    auto T0 = std::chrono::steady_clock::now();
    auto rows = prepared ? load_prepared(csv) : load(csv);
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
    std::fprintf(stderr, "loaded %zu rows in %.1fs\n", rows.size(), load_s);

    std::vector<const Row*> day1, replay, b1, fit_rows, precal;
    if (prepared) {
        for (auto& r : rows) (r.split == 0 ? fit_rows : r.split == 1 ? precal : replay).push_back(&r);
    } else {
    for (auto& r : rows) {
        if (r.ts < DAY2) day1.push_back(&r);
        else if (r.ts < ATTACK_END) replay.push_back(&r);
    }
    for (auto* r : day1) if (r->normal) b1.push_back(r);
    PyRandom rnd(7);
    rnd.shuffle(b1);
    const size_t N_FIT = 10000;
    fit_rows.assign(b1.begin(), b1.begin() + std::min(N_FIT, b1.size()));
    std::unordered_map<const Row*, bool> used;
    for (auto* r : fit_rows) used[r] = true;
    for (auto* r : day1) if (!used.count(r)) precal.push_back(r);
    }

    auto T1 = std::chrono::steady_clock::now();
    std::vector<Request> fit;
    for (auto* r : fit_rows) fit.push_back(to_request(r->raw));
    auto det = std::make_unique<Hybrid>();
    if (std::getenv("SSE_EXACT")) { det->tok.exact = true; det->tok.E.assign(1u << TokenThresholds::EBITS, 0); }
    det->gate.fit(fit);
    det->sem.fit(fit);
    double fit_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - T1).count();
    size_t sem_fit_bytes = det->sem.nbytes();
    std::fprintf(stderr, "fit in %.1fs: gate thr %.6f, semantic thr %.6f, semantic store %.2f MB\n", fit_s,
                 det->gate.thr, det->sem.thr, sem_fit_bytes / 1e6);

    std::string trace;
    std::string* tp = tracep.empty() ? nullptr : &trace;
    LapResult pre = run(*det, precal, "full precal", tp);
    std::fprintf(stderr, "full precal: tpr=%.3f fpr=%.3f wrong=%ld\n", pre.tpr(), pre.fpr(), pre.wrong());
    LapResult l1 = run(*det, replay, "full lap1", tp);
    std::fprintf(stderr, "== full lap 1: tpr=%.4f fpr=%.4f precision=%.4f wrong=%ld (%.1fs, %.3f ms/req)\n", l1.tpr(),
                 l1.fpr(), l1.precision(), l1.wrong(), l1.seconds, l1.seconds / replay.size() * 1e3);
    if (tp) { std::ofstream f(tracep); f << trace; }
    size_t sem_after = det->sem.nbytes();
    long tok_set = det->tok.entries_set(), tok_flipped = det->tok.entries_flipped();
    std::ostringstream gw;
    for (int k = 0; k < NFIELD; k++) gw << (k ? ", " : "") << jstr(FIELD_NAMES[k]) << ": " << jnum(det->gate.w[k]);

    // frozen pass: inference only over the same stream, dist caches cleared, per-request latency
    for (auto& sp : det->sem.space) if (sp) sp->dc.clear();
    std::vector<double> lat;
    lat.reserve(replay.size());
    long ftp = 0, ffp = 0, ffn = 0, ftn = 0;
    auto Tf = std::chrono::steady_clock::now();
    for (auto* r : replay) {
        auto t = std::chrono::steady_clock::now();
        Request q = to_request(r->raw);
        std::string rt = route_of(r->raw);
        bool flagged = det->step(rt, q, !r->normal, false, false, nullptr);
        lat.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count());
        (!r->normal ? (flagged ? ftp : ffn) : (flagged ? ffp : ftn))++;
    }
    double frozen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - Tf).count();
    std::vector<double> sl(lat);
    std::sort(sl.begin(), sl.end());
    auto pct = [&](double p) { return sl[std::min(sl.size() - 1, size_t(p * sl.size()))]; };
    if (std::getenv("SSE_PROFILE")) {                            // where the frozen pass spends its time
        using clk = std::chrono::steady_clock;
        double tr = 0, tg = 0, ts = 0, tt = 0;
        long misses = 0;
        for (auto& sp : det->sem.space) if (sp) sp->dc.clear();
        Space::profiling = true;
        for (auto* r : replay) {
            auto a = clk::now();
            Request q = to_request(r->raw);
            std::string rt = route_of(r->raw);
            auto b = clk::now();
            Hybrid::Feat f;
            det->gate.header_z(q, f.zg);
            f.zg.emplace_back(REQ, det->gate.score(q));
            auto c = clk::now();
            for (auto& [k, v] : q.h) if (det->sem.has_stat[k]) {
                auto& sp = *det->sem.space[k];
                auto it = sp.dc.find(v.s);
                misses += it == sp.dc.end();
            }
            det->sem.header_z(q, f.zs);
            auto d = clk::now();
            std::vector<int> og, os;
            det->over(rt, "gate", f.zg, det->gate.thr, og);
            det->over(rt, "sem", f.zs, det->sem.thr, os);
            auto e = clk::now();
            tr += std::chrono::duration<double, std::micro>(b - a).count();
            tg += std::chrono::duration<double, std::micro>(c - b).count();
            ts += std::chrono::duration<double, std::micro>(d - c).count();
            tt += std::chrono::duration<double, std::micro>(e - d).count();
        }
        Space::profiling = false;
        double nm = std::max(1.0, Space::prof[5]);
        std::fprintf(stderr, "profile us/miss (%.0f misses): weights %.2f, code %.2f, hamming %.2f, select %.2f, rescore %.2f\n",
                     nm, Space::prof[0] / nm, Space::prof[1] / nm, Space::prof[2] / nm, Space::prof[3] / nm, Space::prof[4] / nm);
        double n = double(replay.size());
        std::fprintf(stderr, "profile us/request: parse+analyse %.1f, gate %.1f, semantic %.1f (%.2f cache misses/req), "
                     "token thresholds %.1f\n", tr / n, tg / n, ts / n, misses / n, tt / n);
        size_t tot = 0;
        for (int k = 0; k < NFIELD; k++) if (det->sem.space[k]) {
            std::fprintf(stderr, "  %s: %d stored values\n", FIELD_NAMES[k], det->sem.space[k]->n);
            tot += det->sem.space[k]->n;
        }
        std::fprintf(stderr, "  total %zu\n", tot);
    }
    std::fprintf(stderr, "frozen pass: %.1fs, %.1f us/req mean, p50 %.1f p99 %.1f us, %.0f req/s on one core\n",
                 frozen_s, frozen_s / replay.size() * 1e6, pct(0.5), pct(0.99), replay.size() / frozen_s);

    std::vector<std::string> belt;
    for (int lap = 2; lap <= laps; lap++) {
        std::string name = "full lap" + std::to_string(lap);
        LapResult r = run(*det, replay, name.c_str(), nullptr);
        std::fprintf(stderr, "== full lap %d: tpr=%.4f fpr=%.4f precision=%.4f wrong=%ld (%.0fs)\n", lap, r.tpr(),
                     r.fpr(), r.precision(), r.wrong(), r.seconds);
        belt.push_back("\"" + std::to_string(lap) + "\": " + lap_json(r, replay.size()));
    }

    std::ofstream o(outp);
    o << "{\n \"implementation\": \"cpp/replay_srbh.cpp (C++20, one core)\",\n"
      << " \"protocol\": \"harness/replay_hybrid2.py, HYB_VARIANT=full HYB_QUANT=3: fully binary semantic store, Hamming "
         "top-64 then uint8-count x int8-query int32 rescore and an exact refine of the int8 top 4; packed int8 path-token "
         "thresholds. " << (prepared ? "Splits (fit / precal / test) and order from " + csv
                                     : std::string("Fit on 10,000 normal 17 Jul requests, precalibration on the rest of "
                                                   "17 Jul, lap 1 on 18-19 Jul")) << "; laps 2+ replay the test stream; "
         "teacher = dataset label.\",\n"
      << " \"rows\": " << rows.size() << ", \"load_seconds\": " << jnum(load_s) << ", \"fit_seconds\": " << jnum(fit_s)
      << ",\n \"fit_requests\": " << fit.size() << ", \"precal_requests\": " << precal.size()
      << ", \"replay_requests\": " << replay.size() << ",\n \"gate_thr\": " << jnum(det->gate.thr)
      << ", \"semantic_thr\": " << jnum(det->sem.thr) << ",\n \"precal\": " << lap_json(pre, precal.size())
      << ",\n \"lap1\": " << lap_json(l1, replay.size())
      << ",\n \"frozen_after_lap1\": {\"tpr\": " << jnum(double(ftp) / std::max(1L, ftp + ffn))
      << ", \"fpr\": " << jnum(double(ffp) / std::max(1L, ffp + ftn)) << ", \"seconds\": " << jnum(frozen_s)
      << ", \"us_per_request_mean\": " << jnum(frozen_s / replay.size() * 1e6) << ", \"us_p50\": " << jnum(pct(0.5))
      << ", \"us_p95\": " << jnum(pct(0.95)) << ", \"us_p99\": " << jnum(pct(0.99))
      << ", \"requests_per_second_one_core\": " << jnum(replay.size() / frozen_s) << "}"
      << ",\n \"semantic_bytes_at_fit\": " << sem_fit_bytes << ", \"semantic_bytes_after_lap1\": " << sem_after
      << ",\n \"exact_route_layer\": " << (det->tok.exact ? "true" : "false")
      << ",\n \"route_state\": {\"exact_entries_set\": " << det->tok.exact_set() << ", \"entries_set\": " << tok_set << ", \"entries_flipped\": " << tok_flipped
      << ", \"bytes\": " << det->tok.bytes() << "},\n \"gate_field_weights\": {" << gw.str() << "}";
    if (!belt.empty()) {
        o << ",\n \"belt\": {";
        for (size_t i = 0; i < belt.size(); i++) o << (i ? ", " : "") << belt[i];
        o << "}";
    }
    o << "\n}\n";
    std::fprintf(stderr, "wrote %s (%.0fs)\n", outp.c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count());
    return 0;
}
