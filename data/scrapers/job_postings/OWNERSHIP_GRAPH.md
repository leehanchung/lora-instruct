# Ownership Graph — AI Training & Data Annotation Ecosystem

> **NOTE**: This file is superseded by [SUPPLY_CHAIN.md](SUPPLY_CHAIN.md) which includes
> sourced links, dates, and a full Customer/Supplier taxonomy. This file is kept for
> quick reference only.

## Format

Each triplet: `Subject --[relationship]--> Object`

## Ownership & Subsidiary Relationships

```
# Invisible Technologies (⚠️ Perplexity ownership UNVERIFIED — may be a different "Invisible")
Invisible Technologies --[operates]--> Meridial (expert contractor marketplace)
Invisible Technologies --[acquired Mar 2026]--> WeCP (expert validation platform)

# Surge AI chain
Surge AI               --[operates]--> DataAnnotation.tech (freelancer annotation platform)
Surge AI               --[operates]--> TaskUp.ai (task-based work platform)
Surge AI               --[operates]--> GetHybrid.io (hybrid work platform)

# Scale AI chain
Scale AI               --[operates]--> Outlier AI (expert AI training platform)
Scale AI               --[operates]--> Remotasks (crowdsourced annotation platform)

# Labelbox chain
Labelbox               --[operates]--> Alignerr (AI training workforce platform)

# Nebius / Toloka chain (ex-Yandex)
Nebius Group NV        --[owns]--> Toloka AI (data labeling platform)
Nebius Group NV        --[owns]--> TripleTen (edtech)
Nebius Group NV        --[owns]--> Avride (autonomous driving)
Toloka                 --[operates]--> Mindrift (expert AI training platform)
Yandex N.V.            --[restructured into]--> Nebius Group NV (Jul 2024, sold Russian assets)

# TELUS chain
TELUS Corp             --[owns]--> TELUS Digital (BPO / AI data solutions)
TELUS Digital          --[acquired 2021, $935M]--> Lionbridge AI (1M+ annotators)
TELUS Digital          --[acquired 2021]--> Playment (computer vision annotation)

# Other known corporate parents
Centific               --[subsidiary of]--> Pactera EDGE (IT services)
RWS TrainAI            --[division of]--> RWS Group (language services)
Welocalize             --[independent]--> (language services, AI training division)
Appen                  --[publicly traded]--> ASX:APX (AI data services)
```

## Staffing & Recruiting Relationships

```
NVIDIA                 --[recruits via]--> Sustainable Talent (staffing agency)
```

## End-Client Relationships (who buys annotation/RLHF services)

```
Anthropic              --[contracts]--> Surge AI + DataAnnotation
Anthropic              --[contracts]--> Scale AI + Outlier
Google                 --[contracts]--> Surge AI + DataAnnotation
Google                 --[contracts]--> Scale AI
Google                 --[contracts]--> Nebius Group + Toloka
Meta                   --[contracts]--> Surge AI + DataAnnotation
Meta                   --[contracts]--> Nebius Group + Toloka (Nebius $1.5B Meta deal, 2026)
Microsoft              --[contracts]--> Surge AI + DataAnnotation
OpenAI                 --[contracts]--> Scale AI
```

## Companies That Also Hire RLHF/Training Roles Directly (not yet in dataset)

```
Anthropic              --[hires directly]--> RLHF Specialists, Red Teamers
OpenAI                 --[hires directly]--> Human Data Trainers, Content Evaluators
Google DeepMind        --[hires directly]--> Research Training Roles
Meta FAIR              --[hires directly]--> AI Training / Evaluation Roles
```

## Founder / Leadership

```
Surge AI               --[founded by]--> Edwin Chen (CEO, bootstrapped to $1B+ ARR)
Scale AI               --[founded by]--> Alexandr Wang (CEO)
Invisible Technologies --[founded by]--> Francis Pedraza
Labelbox               --[founded by]--> Manu Sharma (CEO)
Perplexity             --[founded by]--> Aravind Srinivas (CEO)
Nebius Group           --[founded by]--> Arkady Volozh (ex-Yandex founder)
Toloka                 --[CEO]--> Olga Megorskaya
```

## Funding & Valuation

```
Surge AI               --[valuation]--> $1B+ ARR (bootstrapped, no VC)
Scale AI               --[valuation]--> $13.8B (Series F, 2024)
Invisible Technologies --[valuation]--> $2B+ ($100M raise Sep 2025, led by Vanara Capital)
Labelbox               --[valuation]--> $2B+ (Series E, 2024)
Perplexity             --[valuation]--> $9B+ (2025)
Nebius Group           --[market cap]--> ~$10B+ (NASDAQ: NBIS)
Toloka                 --[funding]--> $72M (May 2025, led by Bezos Expeditions)
TELUS Digital          --[acquired Lionbridge AI for]--> $935M (2021)
Appen                  --[market cap]--> ~$300M (ASX: APX, down from $5B peak)
```

## Platform Hierarchy (Annotation Labor Supply Chain)

```
┌──────────────────────────────────────────────────────────────┐
│  END CLIENTS (AI Labs that consume human feedback)            │
│  Anthropic, Google, Meta, Microsoft, OpenAI, NVIDIA, xAI     │
└──────────────┬───────────────────────────────────────────────┘
               │ contracts / hires directly
               ▼
┌──────────────────────────────────────────────────────────────┐
│  HOLDING / PARENT COMPANIES                                   │
│  Perplexity, Nebius Group, TELUS Corp                         │
└──────────────┬───────────────────────────────────────────────┘
               │ owns
               ▼
┌──────────────────────────────────────────────────────────────┐
│  DATA INFRA / ANNOTATION PROVIDERS                            │
│  Surge AI, Scale AI, Labelbox, Invisible Tech, Toloka,        │
│  TELUS Digital, Appen, Sama, iMerit, Innodata, SuperAnnotate  │
└──────────────┬───────────────────────────────────────────────┘
               │ operates
               ▼
┌──────────────────────────────────────────────────────────────┐
│  WORKFORCE PLATFORMS (where freelancers sign up)              │
│  DataAnnotation, Outlier, Remotasks, Alignerr, Meridial,      │
│  Mindrift, TaskUp, GetHybrid, Lionbridge AI                   │
└──────────────┬───────────────────────────────────────────────┘
               │ hires
               ▼
┌──────────────────────────────────────────────────────────────┐
│  FREELANCERS / DOMAIN EXPERTS                                 │
│  700K+ on Outlier, 1M+ ex-Lionbridge, 100K+ DataAnnotation   │
└──────────────────────────────────────────────────────────────┘
```

## Job Count by Entity (from dataset)

| Parent | Platform | Jobs | Relationship |
|--------|----------|------|-------------|
| Labelbox | Alignerr | 48 | subsidiary |
| Surge AI | DataAnnotation | 36 | subsidiary |
| Surge AI | (direct) | 2 | — |
| Scale AI | Outlier | 23 | subsidiary |
| Scale AI | Remotasks | 4 | subsidiary |
| Scale AI | (direct) | 1 | — |
| Perplexity → Invisible Technologies | Meridial | 19 | subsidiary |
| Perplexity → Invisible Technologies | (direct) | 6 | — |
| NVIDIA | Sustainable Talent | 4 | staffing agency |
| Nebius Group | Toloka | 1 | subsidiary |
| Toloka | Mindrift | 2 | subsidiary |
| TELUS Corp → TELUS Digital | (direct) | 2 | — |

## Standalone Companies (no parent mapped yet)

xAI (48), BUKI (6), Innodata (4), YO IT Consulting (3), Taskify AI (3),
SuperAnnotate (2), HumanSignal (2), Anuttacon (2), OpenTrain AI (2),
iMerit (2), Comrise (2), Centific (2), Prolific (1), Juji (1),
Handshake (1), Mercor (1), Collide Capital (1), CloudDevs (1), LXT (1),
Welocalize (1), RWS TrainAI (1), Appen (1), Gloz (1), Embedding VC (1),
Recruiting from Scratch (1), Braintrust (1), Sama (1), CloudFactory (1),
Babel Audio (1), Odixcity Consulting (1), Neon (1)

## Known Companies NOT Yet in Dataset (gaps)

| Company | Why Notable | Status |
|---------|------------|--------|
| Defined.ai | Ethical data annotation, enterprise-grade | No job listings found |
| Encord | Data labeling platform for GenAI, RLHF | Platform, not workforce |
| Cohere | Hires RLHF trainers directly | Not yet scraped |
| Anthropic | Hires RLHF/red-team directly | Not yet scraped |
| OpenAI | Hires human data trainers directly | Not yet scraped |
| Google DeepMind | Research training roles | Not yet scraped |
| Meta FAIR | AI training/evaluation roles | Not yet scraped |
| Snorkel AI | Data labeling platform | Platform, not workforce |
| Labellerr | Annotation platform | Platform, not workforce |
| Kili Technology | Data labeling for ML | Platform, not workforce |

---
*Auto-generated and manually researched. Last updated: 2026-04-11*
