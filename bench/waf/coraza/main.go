// coraza: run every request of a benchmark stream through Coraza v3 + OWASP CRS v4 (coraza-coreruleset) and
// print the inbound anomaly score CRS assigns it (one line per request: score, number of CRS rules matched).
// Detection only, like modsec_eval; same length-prefixed request records (bench/waf/export_requests.py).
//
// usage: coraza <paranoia level 1-4> <requests.bin> <out.txt>
package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"

	coreruleset "github.com/corazawaf/coraza-coreruleset/v4"
	"github.com/corazawaf/coraza/v3"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Fprintln(os.Stderr, "usage: coraza <paranoia 1-4> <requests.bin> <out.txt>")
		os.Exit(2)
	}
	pl := os.Args[1]
	directives := `
SecRuleEngine DetectionOnly
SecRequestBodyAccess On
SecRequestBodyLimit 13107200
SecRequestBodyInMemoryLimit 13107200
SecRequestBodyLimitAction ProcessPartial
SecResponseBodyAccess Off
SecRule REQUEST_HEADERS:Content-Type "^(?:application(?:/soap\+|/)|text/)xml" "id:200000,phase:1,t:none,t:lowercase,pass,nolog,ctl:requestBodyProcessor=XML"
SecRule REQUEST_HEADERS:Content-Type "^application/json" "id:200001,phase:1,t:none,t:lowercase,pass,nolog,ctl:requestBodyProcessor=JSON"
SecAction "id:900000,phase:1,nolog,pass,t:none,setvar:tx.blocking_paranoia_level=` + pl + `"
Include @crs-setup.conf.example
Include @owasp_crs/REQUEST-*.conf
SecAction "id:9999001,phase:2,pass,log,t:none,msg:'SSEBENCH %{tx.blocking_inbound_anomaly_score}'"
`
	waf, err := coraza.NewWAF(coraza.NewWAFConfig().WithRootFS(coreruleset.FS).WithDirectives(directives))
	if err != nil {
		fmt.Fprintln(os.Stderr, "waf:", err)
		os.Exit(1)
	}
	in, err := os.Open(os.Args[2])
	if err != nil {
		panic(err)
	}
	r := bufio.NewReaderSize(in, 1<<20)
	outf, err := os.Create(os.Args[3])
	if err != nil {
		panic(err)
	}
	w := bufio.NewWriter(outf)
	defer w.Flush()
	u32 := func() (uint32, error) {
		var b [4]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			return 0, err
		}
		return binary.LittleEndian.Uint32(b[:]), nil
	}
	t0 := time.Now()
	count := 0
	for {
		n, err := u32()
		if err != nil {
			break
		}
		f := make([]string, n)
		for i := range f {
			l, _ := u32()
			b := make([]byte, l)
			io.ReadFull(r, b)
			f[i] = string(b)
		}
		tx := waf.NewTransaction()
		tx.ProcessConnection("192.0.2.10", 40000, "192.0.2.1", 80)
		tx.ProcessURI(f[1], f[0], "HTTP/1.1")
		for i := 3; i+1 < len(f); i += 2 {
			tx.AddRequestHeader(f[i], f[i+1])
			if strings.EqualFold(f[i], "Host") {
				tx.SetServerName(f[i+1])
			}
		}
		tx.ProcessRequestHeaders()
		if f[2] != "" {
			tx.WriteRequestBody([]byte(f[2]))
		}
		tx.ProcessRequestBody()
		score, matched := -1, 0
		for _, m := range tx.MatchedRules() {
			id := m.Rule().ID()
			if id == 9999001 {
				msg := m.Message()
				if p := strings.Index(msg, "SSEBENCH "); p >= 0 {
					score, _ = strconv.Atoi(strings.TrimSpace(msg[p+9:]))
				}
			} else if id >= 911000 && id < 949000 && m.Message() != "" {
				matched++
			}
		}
		tx.ProcessLogging()
		tx.Close()
		fmt.Fprintf(w, "%d %d\n", score, matched)
		count++
		if count%20000 == 0 {
			fmt.Fprintf(os.Stderr, "  %d requests, %.0f req/s\n", count, float64(count)/time.Since(t0).Seconds())
		}
	}
	s := time.Since(t0).Seconds()
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	fmt.Fprintf(os.Stderr, "%d requests in %.1fs (%.3f ms/request), peak RSS %d kB\n", count, s, s/float64(count)*1e3, ru.Maxrss)
}
