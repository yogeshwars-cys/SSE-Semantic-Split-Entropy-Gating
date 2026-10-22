// modsec_eval.cpp: run every request of a benchmark stream through libmodsecurity v3 + OWASP CRS and print the
// inbound anomaly score CRS assigns it (one line per request: score, number of CRS rules matched).
// The engine runs DetectionOnly so every rule is evaluated; a request is blocked at the default CRS setting when
// its score reaches the inbound threshold (5). The score comes from a rule after REQUEST-949 (see crs.conf.in).
//
// usage: modsec_eval <rules.conf> <requests.bin> <out.txt>
#include <modsecurity/modsecurity.h>
#include <modsecurity/rule_message.h>
#include <modsecurity/rules_set.h>
#include <modsecurity/transaction.h>

#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Ctx { int score = -1, matched = 0; };

static void log_cb(void* data, const void* msg) {
    if (!data || !msg) return;
    auto* ctx = static_cast<Ctx*>(data);
    auto* rm = static_cast<const modsecurity::RuleMessage*>(msg);
    if (rm->m_ruleId == 9999001) {
        const std::string& m = rm->m_message;
        auto p = m.find("SSEBENCH ");
        ctx->score = p == std::string::npos ? 0 : std::atoi(m.c_str() + p + 9);
    } else if (rm->m_ruleId >= 911000 && rm->m_ruleId < 949000) {
        ctx->matched++;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: modsec_eval <rules.conf> <requests.bin> <out.txt>\n"; return 2; }
    auto* ms = new modsecurity::ModSecurity();
    ms->setServerLogCb(log_cb, modsecurity::RuleMessageLogProperty);
    auto* rules = new modsecurity::RulesSet();
    if (rules->loadFromUri(argv[1]) < 0) { std::cerr << "rules: " << rules->getParserError() << "\n"; return 1; }
    std::ifstream in(argv[2], std::ios::binary);
    std::ofstream out(argv[3]);
    auto u32 = [&](uint32_t& v) { return bool(in.read(reinterpret_cast<char*>(&v), 4)); };
    uint32_t n;
    long count = 0;
    auto t0 = std::chrono::steady_clock::now();
    while (u32(n)) {
        std::vector<std::string> f(n);
        for (auto& s : f) {
            uint32_t len;
            u32(len);
            s.resize(len);
            in.read(s.data(), len);
        }
        Ctx ctx;
        modsecurity::Transaction t(ms, rules, &ctx);
        t.processConnection("192.0.2.10", 40000, "192.0.2.1", 80);
        t.processURI(f[1].c_str(), f[0].c_str(), "1.1");
        for (size_t i = 3; i + 1 < f.size(); i += 2)
            t.addRequestHeader(f[i], f[i + 1]);
        t.processRequestHeaders();
        if (!f[2].empty()) t.appendRequestBody(reinterpret_cast<const unsigned char*>(f[2].data()), f[2].size());
        t.processRequestBody();
        t.processLogging();
        out << ctx.score << " " << ctx.matched << "\n";
        if (++count % 20000 == 0) {
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "  %ld requests, %.0f req/s\n", count, count / s);
        }
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    std::fprintf(stderr, "%ld requests in %.1fs (%.3f ms/request), peak RSS %ld kB\n", count, s, s / count * 1e3,
                 ru.ru_maxrss);
    delete rules;
    delete ms;
    return 0;
}
