# The Hosting Dividend: Do the Olympics and the World Cup Pay Off?

Governments bid billions to host mega-events, promising a lasting boost to growth, tourism and global visibility. Economists are sceptical, and the ex-post studies commissioned by host cities are rarely independent. This repo asks a simple counterfactual question for **11 hosts, 2002–2022**, including India's Delhi Commonwealth Games (2010):

> *How would the host's economy have evolved if it had never been awarded the event?*

## Data
World Bank WDI API, pulled live:
- GDP per capita, constant 2015 US$ (`NY.GDP.PCAP.KD`), 1980 onwards
- International tourist arrivals (`ST.INT.ARVL`)

**Donor pool:** every country with more than 1 million people that has *not* hosted or been awarded a comparable event between 1988 and 2034, with a complete series over the window.

| Host | Event | Awarded | Held |
|---|---|---|---|
| South Korea | FIFA World Cup (co-host) | 1996 | 2002 |
| Greece | Athens Olympics | 1997 | 2004 |
| Germany | FIFA World Cup | 2000 | 2006 |
| China | Beijing Olympics | 2001 | 2008 |
| South Africa | FIFA World Cup | 2004 | 2010 |
| India | Delhi Commonwealth Games | 2003 | 2010 |
| UK | London Olympics | 2005 | 2012 |
| Brazil | World Cup 2014 + Rio Olympics 2016 | 2007 | 2014 |
| Russia | World Cup 2018 + Sochi Winter Olympics 2014 | 2007 | 2014 |
| Japan | Tokyo Olympics | 2013 | 2021 |
| Qatar | FIFA World Cup | 2010 | 2022 |

## Method
Each series is indexed to the **award year**, because spending starts when the bid is won, not when the event opens. Two counterfactuals are built for each host and outcome:

1. **Synthetic control** (Abadie, Diamond & Hainmueller 2010). A weighted average of donor countries, with non-negative weights summing to one, chosen to match the host's 10 pre-award years.
2. **Generalised synthetic control / interactive fixed effects** (Xu 2017), a machine-learning factor model:
   `Y_it = u_i + v_t + λ_i′ f_t + ε_it`
   Latent global factors `f_t` are learned from donors by SVD. The host's loadings `λ_i` are fitted on its pre-period, and the **number of factors is chosen by leave-one-year-out cross-validation**. Unlike classic synthetic control, it can follow a host that is growing faster than every donor (e.g. China).

**Inference: placebo in space.** Every donor is treated as if it had hosted, and its post-/pre-period fit ratio is computed. The host's p-value is its rank in that distribution.

## Results
<!-- RESULTS:START -->
_Results, tables and figures are generated automatically by the GitHub Action (`Actions` tab → **Run analysis**). They appear here a few minutes after the first push._
<!-- RESULTS:END -->

## Reproduce
```bash
pip install -r requirements.txt
python run.py
```
The GitHub Action pulls WDI and re-runs everything on push and monthly.

## Limitations
- GDP is a blunt measure. Effects on a host *city* (Delhi, London) are diluted at national level.
- Some hosts overlap with big shocks: the 2008 financial crisis, Greece's debt crisis, sanctions on Russia after 2014, COVID in Tokyo 2021. The placebo p-values show how unusual a host's path is, but cannot attribute it to the event alone.
- Tourism arrivals have a shorter series and COVID truncates the post-period for recent hosts.

## References
- Abadie, A., Diamond, A. & Hainmueller, J. (2010). *Synthetic control methods for comparative case studies.* JASA.
- Xu, Y. (2017). *Generalized synthetic control method.* Political Analysis.
- Baade, R. & Matheson, V. (2016). *Going for the gold: the economics of the Olympics.* Journal of Economic Perspectives.
