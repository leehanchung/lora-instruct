# AI Data Annotation & RLHF Supply Chain

Supply chain mapping for the AI training data industry: who buys annotation/RLHF services
(Customers) and who provides them (Suppliers), with sourced proof for each relationship.

Last updated: 2026-04-11

---

## CUSTOMERS (AI Labs / Model Builders)

Companies that **consume** human annotation, RLHF, red-teaming, and evaluation data
to train their foundation models.

### Anthropic

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Customer — buys RLHF/annotation data | — | — |
| Supplier: Surge AI | Anthropic uses Surge AI's RLHF platform to train Claude | [Surge AI Blog](https://surgehq.ai/blog/anthropic-surge-ai-rlhf-platform-train-llm-assistant-human-feedback) | 2023 |
| Supplier: Scale AI | Anthropic worked with Scale AI for annotation | [Interconnects / Nathan Lambert](https://www.interconnects.ai/p/alignment-as-a-service) | 2023 |
| Headcount | ~4,585 employees by Feb 2026 | [Metaintro](https://www.metaintro.com/blog/ai-trainer-jobs-150-percent-growth-tech-companies-hiring-2026) | Feb 2026 |
| Direct hiring | Hires RLHF specialists and red teamers directly | [Anthropic Careers](https://www.anthropic.com/careers) | Ongoing |

### Meta

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Customer — buys annotation data; also investor in Scale AI | — | — |
| Supplier: Scale AI | Meta acquired 49% of Scale AI for $14.3B; Alexandr Wang moved to Meta as Chief AI Officer | [CNBC](https://www.cnbc.com/2025/06/12/scale-ai-founder-wang-announces-exit-for-meta-part-of-14-billion-deal.html) | Jun 2025 |
| Supplier: Scale AI | Deal values Scale AI at $29B; Meta has no voting power | [TechCrunch](https://techcrunch.com/2025/06/13/new-details-emerge-on-metas-14-3b-deal-for-scale/) | Jun 2025 |
| Supplier: Surge AI | Meta is among Surge AI's ~12 frontier lab clients | [Forbes via Techmeme](https://www.techmeme.com/250921/p5) | Sep 2025 |
| Supplier: Nebius Group | Meta signed $27B AI infra deal with Nebius (compute, not annotation) | [CNBC](https://www.cnbc.com/2026/03/16/meta-nebius-ai-infrastructure.html) | Mar 2026 |
| Supplier: Toloka (Nebius) | Meta uses Toloka for data generation at scale | [Nebius Blog](https://nebius.com/blog/posts/digest-november-2025) | Nov 2025 |
| Note | After Meta's Scale AI investment, OpenAI and Google reduced/paused Scale contracts | [Gun.io](https://gun.io/news/2025/12/scale-ai-alternatives-for-enterprise-ai-teams/) | Dec 2025 |

### OpenAI

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Customer — buys RLHF/annotation data | — | — |
| Supplier: Scale AI | Scale AI was OpenAI's "preferred partner" for GPT-3.5 fine-tuning (Aug 2023); relationship reduced after Meta investment | [Scale AI Wikipedia](https://en.wikipedia.org/wiki/Scale_AI) | Aug 2023 |
| Supplier: Surge AI | OpenAI uses Surge; co-built GSM8K math dataset with Surge | [Sacra](https://sacra.com/c/surge-ai/) | 2022-2024 |
| Headcount | Targeting 8,000 employees by end of 2026 | [Metaintro](https://www.metaintro.com/blog/ai-trainer-jobs-150-percent-growth-tech-companies-hiring-2026) | 2026 |
| Direct hiring | Hires human data trainers and content evaluators directly | [HeroHunt Guide](https://www.herohunt.ai/blog/how-ai-labs-are-hiring-people-to-train-models-2026-insider-guide) | 2026 |

### Google / DeepMind

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Customer — buys annotation data | — | — |
| Supplier: Surge AI | Google among Surge AI's ~12 frontier lab clients | [Forbes via Techmeme](https://www.techmeme.com/250921/p5) | Sep 2025 |
| Supplier: Scale AI | Google reduced Scale AI contracts after Meta acquisition | [Gun.io](https://gun.io/news/2025/12/scale-ai-alternatives-for-enterprise-ai-teams/) | 2025 |
| Supplier: Appen | Google terminated $82.8M annual Appen contract (Mar 2024), ~30% of Appen revenue | [Multiple sources](https://gun.io/news/2025/12/scale-ai-alternatives-for-enterprise-ai-teams/) | Mar 2024 |
| Supplier: Toloka (Nebius) | Google is a Toloka client | [Toloka AI](https://toloka.ai/) | Ongoing |
| Direct hiring | DeepMind hires research-oriented training roles directly | [HeroHunt Guide](https://www.herohunt.ai/blog/how-ai-labs-are-hiring-people-to-train-models-2026-insider-guide) | 2026 |

### Microsoft

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Customer — buys annotation data | — | — |
| Supplier: Surge AI | Microsoft among Surge AI's ~12 frontier lab clients | [Forbes via Techmeme](https://www.techmeme.com/250921/p5) | Sep 2025 |
| Supplier: Nebius Group | Microsoft signed multi-billion dollar AI infra deal with Nebius | [Nebius Newsroom](https://nebius.com/newsroom/nebius-announces-multi-billion-dollar-agreement-with-microsoft-for-ai-infrastructure) | 2026 |

### xAI (Elon Musk)

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | **Dual: Customer AND Supplier (self-labeling)** | — | — |
| Self-labeling | xAI runs its own in-house annotation team for Grok | [TechCrunch](https://techcrunch.com/2025/09/13/xai-reportedly-lays-off-500-workers-from-data-annotation-team/) | Sep 2025 |
| Strategy pivot | Cut 500 generalist annotators, pivoting to "specialist AI tutors" in STEM, finance, medicine, safety, law | [TechCrunch](https://techcrunch.com/2025/09/13/xai-reportedly-lays-off-500-workers-from-data-annotation-team/) | Sep 2025 |
| Scale | Had ~1,500-person annotation team pre-layoff; planned 10x surge in specialist tutors | [The AI Insider](https://theaiinsider.tech/2025/09/15/xai-cuts-500-data-annotation-roles-in-strategic-shift-toward-specialist-ai-tutors/) | Sep 2025 |
| Jobs in dataset | 48 listings (all direct/in-house roles) | Internal | 2026 |

### Cohere

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Customer — buys annotation data | — | — |
| Supplier: Scale AI | Scale provided "a sizable portion" of Cohere's first Command instruction-following dataset | [Scale AI Customer Story](https://scale.com/customers/cohere) | 2023-2024 |
| Direct hiring | Cohere hires data annotators directly (Spanish, safety tasks) | [Parallel Jobs](https://www.useparallel.com/app/candidate/job/67071cf2a58f8e43ef00528a) | 2024 |

### NVIDIA

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Customer — buys annotation services via staffing agency | — | — |
| Supplier: Sustainable Talent | NVIDIA recruits annotation engineers through Sustainable Talent (staffing agency) | [Indeed job listings](https://www.indeed.com) | 2025-2026 |
| Jobs in dataset | 4 listings (GenAI Annotation Ops Engineer, Data Annotation Engineer, ML Engineer, AI Safety Engineer) | Internal | 2026 |

---

## SUPPLIERS (Data Annotation / RLHF Providers)

Companies that **provide** human annotation, RLHF, evaluation, and red-teaming services.

### Surge AI

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Supplier — RLHF, annotation, evaluation | — | — |
| Founded | 2020 by Edwin Chen (CEO); ex-Google/Facebook/Twitter data scientist | [Inc.](https://www.inc.com/sam-blum/bootstrapped-to-1-billion-surge-ai-ceo-edwin-chen-on-how-he-did-it/91207937) | 2020 |
| Revenue | $1.2B in 2024 (bootstrapped, no VC, profitable from day one) | [GetLatka](https://getlatka.com/blog/surgeai-revenue-valuation/) | 2024 |
| Valuation | Reportedly raising $1B at ~$30B valuation | [Forbes via Techmeme](https://www.techmeme.com/250921/p5) | Sep 2025 |
| Employees | ~250 (full-time + consultants) | [GetLatka](https://getlatka.com/blog/surgeai-revenue-valuation/) | 2024 |
| Expert network | 50,000+ domain experts | [GetLatka](https://getlatka.com/blog/surgeai-revenue-valuation/) | 2024 |
| Subsidiary: DataAnnotation.tech | Freelancer annotation platform (100K+ experts); $20-60+/hr | [DataAnnotation.tech](https://dataannotation.tech/) | Ongoing |
| Subsidiary: TaskUp.ai | Task-based work platform | [Sacra](https://sacra.com/c/surge-ai/) | — |
| Subsidiary: GetHybrid.io | Hybrid work platform | [Sacra](https://sacra.com/c/surge-ai/) | — |
| Clients | OpenAI, Google, Anthropic, Microsoft, Meta, Mistral, US Air Force | [Forbes via Techmeme](https://www.techmeme.com/250921/p5) | Sep 2025 |
| Jobs in dataset | 38 (36 DataAnnotation + 2 direct Surge) | Internal | 2026 |

### Scale AI

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Supplier — annotation, RLHF, computer vision, GenAI data engine | — | — |
| Founded | 2016 by Alexandr Wang (departed Jun 2025 to Meta as Chief AI Officer) | [CNBC](https://www.cnbc.com/2025/06/12/scale-ai-founder-wang-announces-exit-for-meta-part-of-14-billion-deal.html) | Jun 2025 |
| Revenue | ~$870M in 2024 | [Sacra](https://sacra.com/c/scale-ai/) | 2024 |
| Valuation | $29B (after Meta's $14.3B for 49% stake) | [TechCrunch](https://techcrunch.com/2025/06/13/new-details-emerge-on-metas-14-3b-deal-for-scale/) | Jun 2025 |
| Meta ownership | 49% non-voting stake; Scale remains independent | [CoinCentral](https://coincentral.com/scale-ai-stresses-independence-as-meta-acquires-49-stake/) | Jun 2025 |
| Impact | OpenAI and Google reduced/paused Scale contracts after Meta deal | [Gun.io](https://gun.io/news/2025/12/scale-ai-alternatives-for-enterprise-ai-teams/) | 2025 |
| Subsidiary: Outlier AI | Expert AI training platform (700K+ experts, 40+ domains) | [Outlier.ai](https://outlier.ai/) | Ongoing |
| Subsidiary: Remotasks | Crowdsourced annotation platform (computer vision, NLP, LiDAR) | [Remotasks](https://www.remotasks.com/) | Ongoing |
| Clients | Meta (investor), Cohere, US DoD, Toyota, and formerly OpenAI/Google | [Scale AI Wikipedia](https://en.wikipedia.org/wiki/Scale_AI) | Various |
| Jobs in dataset | 28 (23 Outlier + 4 Remotasks + 1 direct) | Internal | 2026 |

### Labelbox

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Supplier — RLHF platform, annotation, red-teaming | — | — |
| Founded | By Manu Sharma (CEO) | [Labelbox](https://labelbox.com/) | — |
| Funding | $110M Series E; $189M total raised | [SalesTools AI](https://salestools.io/en/report/labelbox-raises-110m-series-e) | 2024 |
| Valuation | $1B+ (unicorn) | [SalesTools AI](https://salestools.io/en/report/labelbox-raises-110m-series-e) | 2024 |
| Lead investors | SoftBank Vision Fund 2, Andreessen Horowitz | [SalesTools AI](https://salestools.io/en/report/labelbox-raises-110m-series-e) | 2024 |
| Revenue | ~$50M in 2024 | [GetLatka](https://getlatka.com/companies/labelbox) | 2024 |
| Subsidiary: Alignerr | AI training workforce platform (freelance experts) | [Alignerr.com](https://www.alignerr.com/) | Ongoing |
| Jobs in dataset | 48 (all Alignerr) | Internal | 2026 |

### Nebius Group / Toloka

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Supplier — AI infrastructure + data labeling (via Toloka) | — | — |
| Parent: Nebius Group NV | Ex-Yandex N.V.; restructured Jul 2024 after selling Russian assets | [TechCrunch](https://techcrunch.com/2024/07/21/from-yandexs-ashes-comes-nebius-a-startup-with-plans-to-be-a-european-ai-compute-leader/) | Jul 2024 |
| Founded by | Arkady Volozh (ex-Yandex founder) | [Nebius Wikipedia](https://en.wikipedia.org/wiki/Nebius_Group) | — |
| Market cap | ~$10B+ (NASDAQ: NBIS) | [Techi](https://www.techi.com/nebius-stock/) | 2026 |
| Subsidiary: Toloka AI | Data labeling and annotation platform; unit of Nebius | [SiliconANGLE](https://siliconangle.com/2025/05/07/ai-data-provider-toloka-raises-72m-funding/) | May 2025 |
| Toloka funding | $72M led by Bezos Expeditions | [SiliconANGLE](https://siliconangle.com/2025/05/07/ai-data-provider-toloka-raises-72m-funding/) | May 2025 |
| Sub-subsidiary: Mindrift | Expert AI training platform; owned/operated by Toloka | [Mindrift About](https://mindrift.ai/about) | Ongoing |
| Subsidiary: TripleTen | EdTech company | [Nebius Wikipedia](https://en.wikipedia.org/wiki/Nebius_Group) | — |
| Subsidiary: Avride | Autonomous driving | [Nebius Wikipedia](https://en.wikipedia.org/wiki/Nebius_Group) | — |
| Client: Meta | $27B AI infra deal (5-year, compute + capacity) | [CNBC](https://www.cnbc.com/2026/03/16/meta-nebius-ai-infrastructure.html) | Mar 2026 |
| Client: Microsoft | Multi-billion dollar AI infra deal | [Nebius Newsroom](https://nebius.com/newsroom/nebius-announces-multi-billion-dollar-agreement-with-microsoft-for-ai-infrastructure) | 2026 |
| Jobs in dataset | 3 (1 Toloka + 2 Mindrift) | Internal | 2026 |

### Invisible Technologies

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Supplier — AI data annotation, expert marketplace | — | — |
| Founded | 2015 by Francis Pedraza | [BusinessWire](https://www.businesswire.com/news/home/20250916425748/en/Invisible-Technologies-Raises-$100-Million-to-Power-the-Next-Generation-of-AI-Infrastructure-for-the-Enterprise) | 2015 |
| Revenue | $336.3M with 3,100 staff | [GetLatka](https://getlatka.com/companies/invisible-technologies) | 2024 |
| Funding | $100M raise (Sep 2025), $144M total; led by Vanara Capital (TPG spinoff) | [BusinessWire](https://www.businesswire.com/news/home/20250916425748/en/Invisible-Technologies-Raises-$100-Million-to-Power-the-Next-Generation-of-AI-Infrastructure-for-the-Enterprise) | Sep 2025 |
| Valuation | $2B+ | [Bloomberg](https://www.bloomberg.com/news/articles/2025-09-16/scale-ai-rival-invisible-technologies-valued-at-over-2-billion) | Sep 2025 |
| Other investors | Princeville Capital, HOF Capital, Freestyle VC, Acrew Capital, Greycroft | [SiliconANGLE](https://siliconangle.com/2025/09/16/ai-data-provider-invisible-raises-100m-2b-valuation/) | Sep 2025 |
| Subsidiary: Meridial | Expert contractor marketplace (law, STEM, finance, coding, linguistics) | [Meridial.ai](https://www.meridial.ai) | Ongoing |
| Acquisition: WeCP | Expert validation platform for AI training workflows | [BusinessWire](https://www.businesswire.com/news/home/20260310738939/en/Invisible-Technologies-Agrees-to-Acquire-WeCP-to-Strengthen-Expert-Validation-for-High-Precision-AI-Workflows) | Mar 2026 |
| Perplexity link | ⚠️ UNVERIFIED: Some sources say Perplexity acquired an "Invisible" in Aug 2025, but founding details (Minh Pham/JJ Ford, 2023) differ from Invisible Technologies (Francis Pedraza, 2015). May be two different companies. Needs verification. | [Investing.com](https://www.investing.com/news/company-news/perplexity-acquires-invisible-to-boost-ai-agent-infrastructure-93CH-4171144) vs [BusinessWire](https://www.businesswire.com/news/home/20250916425748/en/) | Aug-Sep 2025 |
| Jobs in dataset | 25 (19 Meridial + 6 direct) | Internal | 2026 |

### TELUS Digital (TELUS Corp)

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Supplier — large-scale multilingual annotation, RLHF | — | — |
| Parent: TELUS Corp | Canadian telecom conglomerate | — | — |
| Acquired: Lionbridge AI | $935M (USD); 750+ employees, 1M+ professional annotators, 500+ languages | [BusinessWire](https://www.businesswire.com/news/home/20210302005381/en/TELUS-International-completes-acquisition-of-Lionbridge-AI) | Mar 2021 |
| Acquired: Playment | Bangalore-based; 2D/3D image, video, LiDAR annotation | [BusinessWire](https://www.businesswire.com/news/home/20210706005219/en/TELUS-International-Acquires-Playment-Firmly-Staking-Its-Leadership-in-the-Global-Data-Annotation-Market) | Jul 2021 |
| Client: Top 4 tech | "Supports four of the world's top five technology companies" (unnamed) | [RWS TrainAI](https://www.rws.com/artificial-intelligence/train-ai-data-services/trainai-community/) | 2025 |
| Jobs in dataset | 2 | Internal | 2026 |

### Appen

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Supplier — crowdsourced annotation, multilingual data | — | — |
| Public | ASX: APX (Australian Stock Exchange) | — | — |
| Revenue | $235.7M (after Google contract loss) | [Gun.io](https://gun.io/news/2025/12/scale-ai-alternatives-for-enterprise-ai-teams/) | 2024 |
| Lost client: Google | Google terminated $82.8M annual contract (~30% of Appen revenue) | [Gun.io](https://gun.io/news/2025/12/scale-ai-alternatives-for-enterprise-ai-teams/) | Mar 2024 |
| Market cap | ~$300M (down from $5B peak) | Various | 2025 |
| Jobs in dataset | 1 | Internal | 2026 |

---

## STAFFING AGENCIES (Intermediaries)

| Agency | Client | Roles | Source | Date |
|--------|--------|-------|--------|------|
| Sustainable Talent | NVIDIA | GenAI Annotation Ops Engineer, Data Annotation Engineer, ML Engineer, AI Safety Engineer | [Indeed job listings](https://www.indeed.com) | 2025-2026 |

---

## SUPPLY CHAIN GRAPH (Triplets)

```
# ===== OWNERSHIP / CORPORATE STRUCTURE =====

# Surge AI family
Surge AI               --[operates]--> DataAnnotation.tech        # Source: Sacra, Inc., Forbes
Surge AI               --[operates]--> TaskUp.ai                  # Source: Sacra
Surge AI               --[operates]--> GetHybrid.io               # Source: Sacra

# Scale AI family
Meta                   --[49% stake, $14.3B]--> Scale AI          # Source: CNBC, Jun 2025
Scale AI               --[operates]--> Outlier AI                 # Source: Scale AI / Outlier.ai
Scale AI               --[operates]--> Remotasks                  # Source: Scale AI / Remotasks.com

# Labelbox family
Labelbox               --[operates]--> Alignerr                   # Source: Alignerr.com

# Nebius / Toloka family
Nebius Group NV        --[owns]--> Toloka AI                      # Source: SiliconANGLE, May 2025
Toloka                 --[operates]--> Mindrift                   # Source: Mindrift About page
Yandex N.V.            --[restructured into]--> Nebius Group NV   # Source: TechCrunch, Jul 2024

# Invisible Technologies family
Invisible Technologies --[operates]--> Meridial                   # Source: Meridial.ai, LinkedIn
Invisible Technologies --[acquired]--> WeCP                       # Source: BusinessWire, Mar 2026

# TELUS family
TELUS Corp             --[owns]--> TELUS Digital                  # Source: TELUS.com
TELUS Digital          --[acquired, $935M]--> Lionbridge AI       # Source: BusinessWire, Mar 2021
TELUS Digital          --[acquired]--> Playment                   # Source: BusinessWire, Jul 2021

# Other
Centific               --[subsidiary of]--> Pactera EDGE
RWS TrainAI            --[division of]--> RWS Group


# ===== CUSTOMER → SUPPLIER CONTRACTS =====

# Anthropic buys from:
Anthropic              --[contracts]--> Surge AI                  # Source: Surge AI Blog
Anthropic              --[contracts]--> Scale AI                  # Source: Interconnects

# Meta buys from:
Meta                   --[contracts + owns 49%]--> Scale AI       # Source: CNBC, Jun 2025
Meta                   --[contracts]--> Surge AI                  # Source: Forbes/Techmeme
Meta                   --[contracts, $27B infra]--> Nebius Group  # Source: CNBC, Mar 2026

# OpenAI buys from:
OpenAI                 --[contracts]--> Surge AI                  # Source: Sacra (GSM8K co-built)
OpenAI                 --[formerly contracted]--> Scale AI        # Source: Wikipedia (reduced post-Meta deal)

# Google buys from:
Google                 --[contracts]--> Surge AI                  # Source: Forbes/Techmeme
Google                 --[formerly contracted]--> Scale AI        # Source: Gun.io (reduced post-Meta deal)
Google                 --[terminated, $82.8M]--> Appen            # Source: Gun.io, Mar 2024
Google                 --[contracts]--> Toloka                    # Source: Toloka.ai

# Microsoft buys from:
Microsoft              --[contracts]--> Surge AI                  # Source: Forbes/Techmeme
Microsoft              --[contracts, multi-$B]--> Nebius Group    # Source: Nebius Newsroom

# Cohere buys from:
Cohere                 --[contracts]--> Scale AI                  # Source: Scale AI Customer Story

# xAI (self-supplies):
xAI                    --[self-labeling]--> xAI (in-house)        # Source: TechCrunch, Sep 2025

# NVIDIA uses staffing:
NVIDIA                 --[recruits via]--> Sustainable Talent     # Source: Indeed job listings


# ===== FUNDING / INVESTMENT =====

Surge AI               --[bootstrapped]--> $1.2B revenue (2024)   # Source: GetLatka
Surge AI               --[raising]--> $1B at $30B valuation       # Source: Forbes/Techmeme, Sep 2025
Scale AI               --[valued at]--> $29B (post-Meta deal)     # Source: TechCrunch, Jun 2025
Labelbox               --[Series E]--> $110M at $1B+              # Source: SalesTools AI, 2024
Invisible Technologies --[raised]--> $100M at $2B+                # Source: BusinessWire, Sep 2025
Toloka                 --[raised]--> $72M (Bezos Expeditions)     # Source: SiliconANGLE, May 2025
Nebius Group           --[market cap]--> ~$10B+ (NASDAQ: NBIS)    # Source: Techi, 2026
Appen                  --[market cap]--> ~$300M (ASX: APX)        # Source: Various, down from $5B peak


# ===== LEADERSHIP =====

Surge AI               --[CEO]--> Edwin Chen (bootstrapped, ex-Google/FB/Twitter)
Scale AI               --[former CEO]--> Alexandr Wang (departed to Meta, Jun 2025)
Invisible Technologies --[CEO]--> Francis Pedraza (founded 2015)
Labelbox               --[CEO]--> Manu Sharma
Nebius Group           --[founder]--> Arkady Volozh (ex-Yandex)
Toloka                 --[CEO]--> Olga Megorskaya
```

---

## VERTICAL AI COMPANIES (Build + Train In-House)

Companies that build domain-specific AI products and hire domain experts directly
to train their models — neither pure "customers" of annotation platforms nor "suppliers"
of annotation labor. They represent the emerging "insource the judgment" trend.

### Harvey AI (Legal)

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Vertical AI — builds AI for law firms, hires legal experts directly | — | — |
| Valuation | $1.5B+ | [Forbes](https://www.forbes.com/companies/harvey/) | 2025 |
| Investors | Sequoia Capital, Kleiner Perkins, Google Ventures, OpenAI Startup Fund | [Crunchbase](https://www.crunchbase.com/organization/harvey-ai) | 2024 |
| Job count | 7 Legal Engineer roles ($200-300K + equity) | [Harvey Careers](https://jobs.ashbyhq.com/harvey) | Apr 2026 |
| Requirements | JD from top-tier law school + 3yr at top firm | Harvey job listings | Apr 2026 |
| Coverage | SF, NYC, Dallas, London/EMEA; Tax, Innovation, Custom Solutions specialties | Harvey job listings | Apr 2026 |
| Significance | Highest-paid AI training roles in dataset; represents shift from hourly annotation to full-time equity-compensated legal AI product roles | — | — |

### Hippocratic AI (Healthcare)

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Vertical AI — builds safety-focused healthcare LLMs | — | — |
| Funding | $137M Series A | [TechCrunch](https://techcrunch.com/2024/03/hippocratic-ai-series-a/) | 2024 |
| Investors | General Catalyst, a16z | [Crunchbase](https://www.crunchbase.com/organization/hippocratic-ai) | 2024 |
| Job count | 1 RN/Clinical Product Specialist ($120-180K base + equity) | [Hippocratic AI Careers](https://www.hippocraticai.com/careers) | Apr 2026 |
| Requirements | Active RN license, 3+ years clinical/ICU experience | Hippocratic AI listing | Apr 2026 |
| Significance | First full-time, equity-compensated clinical AI validation role in our dataset | — | — |

---

## ENVIRONMENT BUILDING & EXPERIENCE CURATION

Companies that build RL environments, simulation platforms, and training infrastructure
where AI agents learn. These are upstream infrastructure providers — they don't hire
annotators directly (with a few exceptions) but shape the demand for training data.

### Deeptune

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | RL environment builder — "training gyms" simulating workplace workflows (Slack, Salesforce, DevOps) | — | — |
| Funding | $43M Series A led by Andreessen Horowitz; joined by 776, Abstract Ventures, Inspired Capital | [Fortune](https://fortune.com/2026/03/19/andreessen-horowitz-ai-startups-deeptune-series-a/) | Mar 2026 |
| Seed | $5M seed (prior round) | [FinSMEs](https://www.finsmes.com/2026/03/deeptune-raises-43m-in-series-a-funding.html) | — |
| Team | ~20 people (NYC); engineers from Anthropic, Scale AI, Palantir, Hebbia, Glean, Retool | [Fortune](https://fortune.com/2026/03/19/andreessen-horowitz-ai-startups-deeptune-series-a/) | Mar 2026 |
| Angel investors | Noam Brown (OpenAI), Brendan Foody (Mercor CEO), Yash Patil (Applied Compute) | [Fortune](https://fortune.com/2026/03/19/andreessen-horowitz-ai-startups-deeptune-series-a/) | Mar 2026 |
| Hiring annotation? | No — engineering and ops roles only | [Deeptune Careers](https://jobs.ashbyhq.com/deeptune) | Apr 2026 |

### Mechanize, Inc.

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Virtual work environments + benchmarks for AI agent automation of white-collar jobs | — | — |
| Funding | $100M+ from angels: Nat Friedman, Daniel Gross, Patrick Collison, Dwarkesh Patel, Jeff Dean, Sholto Douglas | [TechCrunch](https://techcrunch.com/2025/04/19/famed-ai-researcher-launches-controversial-startup-to-replace-all-human-workers-everywhere/) | Apr 2025 |
| Founded | Apr 2025 by Tamay Besiroglu, Matthew Barnett, Ege Erdil (all ex-Epoch AI) | [TechCrunch](https://techcrunch.com/2025/04/19/famed-ai-researcher-launches-controversial-startup-to-replace-all-human-workers-everywhere/) | Apr 2025 |
| Mission | "Full automation of all work" — starting with white-collar | [TechCrunch](https://techcrunch.com/2025/04/19/famed-ai-researcher-launches-controversial-startup-to-replace-all-human-workers-everywhere/) | Apr 2025 |
| Hiring annotation? | No — Product Engineers only | [Mechanize Careers](https://jobs.ashbyhq.com/mechanize) | Apr 2026 |

### Bespoke Labs

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Data curation platform — training datasets, eval benchmarks, model fine-tuning | — | — |
| Funding | $7.25M Series A (Jun 2024) | [Tracxn](https://tracxn.com/d/companies/bespoke-labs/__TgeW4_XxZv-sKUrOh6M6QeTLr6e9xHzW26BbTJzHYbQ) | Jun 2024 |
| Founded | 2024 by Alex Dimakis (Mountain View, CA) | [Tracxn](https://tracxn.com/d/companies/bespoke-labs/__TgeW4_XxZv-sKUrOh6M6QeTLr6e9xHzW26BbTJzHYbQ) | 2024 |
| Products | OpenThoughts (reasoning dataset), Bespoke-MiniCheck (factuality model) | [Bespoke Labs](https://www.bespokelabs.ai/) | — |
| Hiring annotation? | **YES — "Human Data for RL" (Contract), $25-40/hr** — design technical problems in data science, DevOps, networking | [Ashby](https://jobs.ashbyhq.com/bespokelabs) | Apr 2026 |

### Sepal AI (acquired by Mercor)

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Training data, eval benchmarks, RL environments for frontier LLMs | — | — |
| Funding (pre-acq) | $500K from Y Combinator, Metaplanet, SID Venture Partners, Sterling Road, Team Ignite | [GetLatka](https://getlatka.com/companies/sepalai.com) | 2024 |
| Revenue (pre-acq) | $2M with 13-person team | [GetLatka](https://getlatka.com/companies/sepalai.com) | 2024 |
| Acquired by | Mercor ($10B valuation, $350M Series C) — acqui-hire, price undisclosed | [Orrick](https://www.orrick.com/en/News/2026/02/Mercor-Acquires-Sepal-AI) | Feb 2026 |
| Mercor valuation | $10B ($350M Series C led by Felicis; Benchmark, General Catalyst) | [TechCrunch](https://techcrunch.com/2025/10/27/mercor-quintuples-valuation-to-10b-with-350m-series-c/) | Oct 2025 |
| Hiring annotation? | **YES — Expert Network (20k+ PhDs, analysts, medical pros) for domain-specific annotation** | [Sepal AI Experts](https://www.sepalai.com/experts) | Apr 2026 |

### Fleet AI

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Simulated worlds + real-world challenges for AI agent behavior shaping | — | — |
| Funding | Backed by Sequoia Capital, Menlo Ventures, Company Ventures, SV Angel (amounts undisclosed) | [Fleet AI](https://www.fleetai.com/) | — |
| Team | Ex-Anthropic, Meta Superintelligence, Microsoft AI, Mercor, Jane Street, Citadel | [Fleet AI Careers](https://www.fleetai.com/careers) | Apr 2026 |
| Hiring annotation? | Partially — **MTS Data** role (data pipelines + scenario design for agent training); not pure annotation | [Fleet AI Careers](https://www.fleetai.com/careers) | Apr 2026 |

### Veris AI

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | Simulation environments for enterprise AI agent testing + training | — | — |
| Funding | $8.5M Seed co-led by Decibel Ventures and Acrew Capital | [BusinessWire](https://www.businesswire.com/news/home/20250603868539/en/) | Jun 2025 |
| Founded | By Mehdi Jamei; HQ in NYC, team in SF + NYC | [BusinessWire](https://www.businesswire.com/news/home/20250603868539/en/) | Jun 2025 |
| Customers | Financial services, enterprise productivity, manufacturing | [Veris AI Blog](https://veris.ai/blog/introducing-veris-ai-a-new-way-to-train-enterprise-ai-agents-through-simulated-experience) | Jun 2025 |
| Hiring annotation? | No — ML Specialist (first ML hire, requires publications) | [Gusto Jobs](https://jobs.gusto.com/boards/veris-technologies-inc-dadc87fc-640c-4bff-837d-fa65c392e997) | Apr 2026 |

### Phinity Labs

| Attribute | Value | Source | Date |
|-----------|-------|--------|------|
| Role | RL environments + expert data labeling for semiconductor/hardware AI (RTL design, verification, P&R) | — | — |
| Funding | $5.5M seed; investors include Jeff Dean (DeepMind) | [Phinity Labs](https://www.phinity.ai/) | 2025 |
| Founded | 2025 in San Francisco; founders trained best open-source frontier models in RTL code gen at NVIDIA | [Phinity Labs](https://www.phinity.ai/) | 2025 |
| Hiring annotation? | **YES (domain expert) — Sr. Founding RTL Design Engineer ($180-240K + 0.1-0.6% equity)** | [Wellfound](https://wellfound.com/jobs/3540724-senior-founding-rtl-design-engineer) | Apr 2026 |

### Others (limited public info)

| Company | What They Do | Funding | Hiring Annotators? |
|---------|-------------|---------|-------------------|
| Preference Model | RL environments (limited info) | Unknown | No public openings |
| Matrices | Training environments for multimodal LLM agents | Unknown | No public openings |
| Habitat Inc. | RL environments for work automation | ~$16M | No public openings |
| OpenReward | Platform for hosting RL environments (Open Reward Standard) | Unknown (built by GR Inc.) | No public openings |
| Vmax | Automates RL with proprietary framework | Stealth; South Park Commons | No (5 people, stealth) |
| Proximal | Training data for frontier AI; multi-agent systems, reward hacking | Unknown | Possible (careers page exists) |
| Steadyworks | Unknown — could not find | Unknown | Unknown |

---

## KEY MARKET EVENTS (Timeline)

| Date | Event | Source |
|------|-------|--------|
| Mar 2021 | TELUS Digital acquires Lionbridge AI for $935M | [BusinessWire](https://www.businesswire.com/news/home/20210302005381/en/) |
| Jul 2021 | TELUS Digital acquires Playment | [BusinessWire](https://www.businesswire.com/news/home/20210706005219/en/) |
| Aug 2023 | Scale AI becomes OpenAI's "preferred partner" for GPT-3.5 fine-tuning | [Scale AI Wikipedia](https://en.wikipedia.org/wiki/Scale_AI) |
| Mar 2024 | Google terminates $82.8M annual Appen contract | [Gun.io](https://gun.io/news/2025/12/scale-ai-alternatives-for-enterprise-ai-teams/) |
| Jul 2024 | Yandex N.V. sells Russian assets, restructures as Nebius Group NV | [TechCrunch](https://techcrunch.com/2024/07/21/from-yandexs-ashes-comes-nebius-a-startup-with-plans-to-be-a-european-ai-compute-leader/) |
| 2024 | Surge AI hits $1.2B revenue (bootstrapped); Scale AI at ~$870M | [GetLatka](https://getlatka.com/blog/surgeai-revenue-valuation/) |
| May 2025 | Toloka raises $72M led by Bezos Expeditions | [SiliconANGLE](https://siliconangle.com/2025/05/07/ai-data-provider-toloka-raises-72m-funding/) |
| Jun 2025 | Meta acquires 49% of Scale AI for $14.3B ($29B valuation); Alexandr Wang departs to Meta as Chief AI Officer | [CNBC](https://www.cnbc.com/2025/06/12/scale-ai-founder-wang-announces-exit-for-meta-part-of-14-billion-deal.html) |
| Jun 2025 | OpenAI and Google reduce/pause Scale AI contracts over competitive data concerns | [Gun.io](https://gun.io/news/2025/12/scale-ai-alternatives-for-enterprise-ai-teams/) |
| Sep 2025 | Invisible Technologies raises $100M at $2B+ valuation (led by Vanara Capital) | [BusinessWire](https://www.businesswire.com/news/home/20250916425748/en/) |
| Sep 2025 | xAI lays off 500 generalist data annotators, pivots to specialist AI tutors | [TechCrunch](https://techcrunch.com/2025/09/13/xai-reportedly-lays-off-500-workers-from-data-annotation-team/) |
| Sep 2025 | Surge AI reportedly raising $1B at $30B valuation | [Forbes via Techmeme](https://www.techmeme.com/250921/p5) |
| Mar 2026 | Meta signs $27B AI infrastructure deal with Nebius (5-year) | [CNBC](https://www.cnbc.com/2026/03/16/meta-nebius-ai-infrastructure.html) |
| Mar 2026 | Invisible Technologies acquires WeCP for expert validation | [BusinessWire](https://www.businesswire.com/news/home/20260310738939/en/) |
| Mar 2026 | Deeptune raises $43M Series A led by a16z for AI "training gyms" | [Fortune](https://fortune.com/2026/03/19/andreessen-horowitz-ai-startups-deeptune-series-a/) |
| Feb 2026 | Mercor ($10B) acquires Sepal AI (training data + RL environments) | [Orrick](https://www.orrick.com/en/News/2026/02/Mercor-Acquires-Sepal-AI) |
| Oct 2025 | Mercor raises $350M Series C at $10B valuation (Felicis, Benchmark, General Catalyst) | [TechCrunch](https://techcrunch.com/2025/10/27/mercor-quintuples-valuation-to-10b-with-350m-series-c/) |
| Jun 2025 | Veris AI raises $8.5M seed for enterprise AI agent simulation (Decibel, Acrew) | [BusinessWire](https://www.businesswire.com/news/home/20250603868539/en/) |
| Apr 2025 | Mechanize launches with $100M+ from Nat Friedman, Daniel Gross, Jeff Dean, Patrick Collison | [TechCrunch](https://techcrunch.com/2025/04/19/famed-ai-researcher-launches-controversial-startup-to-replace-all-human-workers-everywhere/) |

---

## NOTES & CAVEATS

1. **Perplexity / Invisible confusion**: Some sources report Perplexity acquired an "Invisible" company in Aug 2025, but the founding details (Minh Pham/JJ Ford, 2023) differ from Invisible Technologies (Francis Pedraza, 2015). Invisible Technologies separately raised $100M in Sep 2025. These may be two different companies sharing the "Invisible" name. Needs further verification.

2. **Nebius deals are AI infrastructure (compute), not annotation**: The $27B Meta and multi-$B Microsoft deals with Nebius are for GPU cloud/data center capacity, not data labeling. Toloka (Nebius subsidiary) is the annotation arm.

3. **Scale AI post-Meta**: After Meta's acquisition of 49%, several labs (OpenAI, Google) reportedly reduced engagement with Scale over data exposure concerns. Scale stresses it remains independent with no Meta voting power or data access.

4. **Spending estimates**: Leading AI labs reportedly spend ~$1B/year each on human training data (collective industry estimate).

---
*Compiled from web research. All sources linked inline. Last updated: 2026-04-11*
