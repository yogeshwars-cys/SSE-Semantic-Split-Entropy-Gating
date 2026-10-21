"""sites.py: the five fake `.demo` domains and the toy search engine the origin serves.

Everything here is make-believe for a closed-lab demo: the hosts resolve nowhere, the "bank" and "clinic" hold no
real data, and the only thing an attack can reach is our own cache. Each site keeps the same handful of poisonable
inputs as the single shop did (BENCHMARKS.md Part I), so the split check still only fires on real poison:

  x-forwarded-host     becomes the host of the page's <script src>
  x-forwarded-prefix   becomes the stylesheet's path prefix
  accept-language      its first tag is written into <html lang="..."> unescaped
  utm_content          is echoed into the page's share link (the cache leaves utm_* out of the key)
  user-agent           a legacy MSIE gets the "browser not supported" page
  cookie               over 4 KB is answered 400, and the error is cacheable (CPDoS)

Benign variation in any of these leaves the page byte-identical, so a clean re-fetch (split's Part B) matches.
"""

SEARCH_HOST = "find.demo"

# host -> brand, theme colour, tagline, and its pages/products. Colours drive the dashboards' per-site chips.
SITES = {
    "kestrel.demo": {
        "brand": "Kestrel Supply", "colour": "#b45309", "kind": "shop",
        "tagline": "Field gear that lasts, repaired free for ten years.",
        "pages": {
            "/": ("Field gear that lasts", "Waxed canvas packs, steel flasks and wool layers."),
            "/blog": ("Workshop notes", "How we re-wax a 2014 rucksack, and why we dropped nylon thread."),
        },
        "products": {
            "1": ("Ridgeline 28L pack", "Waxed canvas, leather straps, 1.4 kg. 189 EUR."),
            "2": ("Ember steel flask", "Double-walled, 750 ml, hot for 14 hours. 42 EUR."),
            "3": ("Moor wool overshirt", "Boiled merino, three sizes, made in Porto. 128 EUR."),
        },
    },
    "aurora.demo": {
        "brand": "Aurora Daily", "colour": "#1d4ed8", "kind": "news",
        "tagline": "Independent reporting from the northern coast.",
        "pages": {
            "/": ("Today's edition", "Harbour works overrun; a profile of the last sail-maker in town."),
            "/blog": ("Opinion", "Why the ferry timetable is a climate policy in disguise."),
        },
        "products": {
            "1": ("Harbour works overrun", "The council's pier repair is two years late and 40% over budget."),
            "2": ("The last sail-maker", "At 71, Ana Reyes still cuts canvas by hand above the old dock."),
            "3": ("Ferry or bridge?", "Three readings of the strait crossing the region keeps deferring."),
        },
    },
    "vaultline.demo": {
        "brand": "Vaultline Bank", "colour": "#047857", "kind": "bank",
        "tagline": "A pretend bank for a security demo. No real accounts.",
        "pages": {
            "/": ("Everyday banking", "Current accounts, cards and transfers. (Demo only; nothing is real.)"),
            "/blog": ("Security notices", "We will never ask for your PIN. This whole site is a lab mock-up."),
        },
        "products": {
            "1": ("Current account", "No monthly fee, instant transfers. Representative demo figures."),
            "2": ("Travel card", "Spend in 140 currencies at the mid-market rate. Demo product."),
            "3": ("Savings pot", "2.1% on balances up to 20,000. Illustrative only."),
        },
    },
    "medina.demo": {
        "brand": "Medina Clinic", "colour": "#7c3aed", "kind": "clinic",
        "tagline": "A pretend clinic for a security demo. No real patients.",
        "pages": {
            "/": ("Book an appointment", "GPs, physiotherapy and travel vaccines. (Demo only.)"),
            "/blog": ("Health notes", "Flu-jab season and how to prepare. Illustrative content only."),
        },
        "products": {
            "1": ("General practice", "Same-week appointments with a named GP. Demo service."),
            "2": ("Physiotherapy", "45-minute assessment and a home plan. Demo service."),
            "3": ("Travel clinic", "Vaccines and advice for 60 countries. Demo service."),
        },
    },
    "civic.demo": {
        "brand": "Civic Portal", "colour": "#be123c", "kind": "city",
        "tagline": "A pretend city-services site for a security demo.",
        "pages": {
            "/": ("City services", "Bins, parking permits and council tax. (Demo only.)"),
            "/blog": ("Announcements", "Road closures this month and the new recycling scheme."),
        },
        "products": {
            "1": ("Parking permit", "Resident permits by zone, renewed online. Demo service."),
            "2": ("Bulky waste", "Book a kerbside collection for large items. Demo service."),
            "3": ("Council tax", "Bands, discounts and how to pay. Demo information."),
        },
    },
}


def catalogue():
    """[{host, brand, colour, kind}] for the dashboards and the client's host picker."""
    out = [{"host": h, "brand": s["brand"], "colour": s["colour"], "kind": s["kind"]} for h, s in SITES.items()]
    out.append({"host": SEARCH_HOST, "brand": "Find.demo", "colour": "#0f766e", "kind": "search"})
    return out


def hosts():
    return list(SITES) + [SEARCH_HOST]
