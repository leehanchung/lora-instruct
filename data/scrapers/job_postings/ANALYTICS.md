# AI Training Job Market — Sector Analytics

Analysis of 262 AI trainer / data annotation job postings classified by service sector,
using the framework from Sequoia Capital's ["Services: The New Software"](https://sequoiacap.com/article/services-the-new-software/) (Mar 2026).

Sequoia's thesis: for every $1 spent on software, $6 is spent on services. AI can automate
the "intelligence" portion of services (rule-following, pattern-matching) while the "judgment"
portion (experience, taste, domain expertise) remains human. The article maps service verticals
on an intelligence↔judgment spectrum against an outsourced↔insourced axis to identify where
AI-powered service companies will emerge first.

We apply this lens to the AI annotation/RLHF labor market itself — these jobs *are* the
human-intelligence layer that trains the models.

Last updated: 2026-04-11 | Source: 262 jobs across 41 parent companies (254 Annotation/Contract + 8 Corporate/In-House)

> **Category note:** Most jobs (254) are freelance/contract annotation and RLHF roles. Eight jobs
> (7 Harvey AI, 1 Hippocratic AI) are tagged **Corporate/In-House** — full-time salaried positions
> at vertical AI companies where domain experts are embedded in the product team. These are included
> for market intelligence but represent a qualitatively different labor model (equity-compensated
> product roles vs hourly annotation work).

---

## Investment Density: Annotation Jobs vs Market Size

This table maps our observed annotation job openings against two market size estimates:
(1) Sequoia's service vertical TAM (total outsourced labor spend) and
(2) the AI training data sub-market for that vertical. The **investment signal** gauges
whether hiring intensity matches the opportunity size.

| Sector | Jobs | % | Sequoia Services TAM | AI Training Data TAM | Median Pay | Investment Signal |
|--------|-----:|---:|---------------------:|---------------------:|-----------:|-------------------|
| Content & Creative | 51 | 20.8% | $300-400B (consulting + media) | $1.0-1.5B | $40-65/hr | ACTIVE — high volume, well-funded |
| STEM & Research | 32 | 13.1% | $100+B (R&D services) | $0.5-0.8B | $40-65/hr | HOT — premium pay, growing fast |
| Education & Training | 28 | 11.4% | $150+B (US education) | $0.3-0.5B | $20-35/hr | ACTIVE — xAI bet driving demand |
| IT & Data Services | 24 | 9.8% | $100+B (IT managed) | $1.2-1.5B | $10-12/hr | SHRINKING — automation replacing base |
| Software Engineering | 24 | 9.8% | $200+B (dev services) | $0.8-1.2B | $40-60/hr | HOT — code eval is highest-value RLHF |
| Accounting & Finance | 22 | 9.0% | $50-80B (US outsourced) | $0.3-0.4B | $30-60/hr | ACTIVE — CPA shortage fueling urgency |
| General AI Training | 19 | 7.8% | N/A | N/A | $14-65/hr | DECLINING — specialists replacing generalists |
| Healthcare & Life Sci | 19 (18 annotation + 1 corporate) | 7.3% | $50-80B (US outsourced) | $0.8-1.0B | $40-60/hr | UNDERINVESTED — 18% of AI data market, only 7% of jobs; Hippocratic AI ($137M) entering |
| Engineering & Industrial | 10 | 4.1% | $150+B (eng services) | $1.0-1.3B | $30-60/hr | UNDERINVESTED — 29% of annotation market, only 4% of jobs |
| Language & Localization | 9 | 3.7% | $72B (language services) | $0.3-0.5B | $20-40/hr | UNDERINVESTED — huge TAM, few openings |
| Legal Services | 14 (7 annotation + 7 corporate) | 5.3% | $60B (Sequoia est.) | $0.1-0.2B | $45/hr (annot.) · $200-300K (corp.) | HEATING UP — Harvey AI ($200-300K corporate) raising the bar; annotation side still thin |
| Business & Digital + Safety | 2 | 0.8% | $50+B (digital + cyber) | <$0.1B | $30-55/hr | EMERGING — nascent, expect rapid growth |

### TAM Sources

| Vertical | Sequoia TAM Source | AI Training TAM Source |
|----------|-------------------|----------------------|
| Content & Creative | [Sequoia: Management Consulting $300-400B](https://sequoiacap.com/article/services-the-new-software/) | [Grand View Research](https://www.grandviewresearch.com/industry-analysis/ai-training-dataset-market) — NLP/text = ~25-30% of $4.4B market |
| STEM & Research | [Sequoia: R&D services](https://sequoiacap.com/article/services-the-new-software/) | Estimated from healthcare + IT segment shares |
| Education & Training | US Dept of Education; [Sequoia](https://sequoiacap.com/article/services-the-new-software/) | Estimated from market growth reports |
| IT & Data Services | [Sequoia: IT Managed Services $100+B](https://sequoiacap.com/article/services-the-new-software/) | [Business Research Insights](https://www.businessresearchinsights.com/market-reports/ai-training-dataset-market-110110) — IT/telecom = 27% of market |
| Software Engineering | [Sequoia: lead example in article](https://sequoiacap.com/article/services-the-new-software/) | Estimated from coding trainer premium pricing |
| Accounting & Finance | [Sequoia: $50-80B US outsourced](https://sequoiacap.com/article/services-the-new-software/) | [Scoop Market.us](https://scoop.market.us/ai-training-dataset-statistics/) — BFSI = 6% of market |
| Healthcare & Life Sci | [Sequoia: $50-80B US outsourced](https://sequoiacap.com/article/services-the-new-software/) | [Precedence Research](https://www.precedenceresearch.com/ai-annotation-market) — healthcare = 18%, CAGR 29.7% |
| Engineering & Industrial | [Sequoia: engineering services](https://sequoiacap.com/article/services-the-new-software/) | [IMARC Group](https://www.imarcgroup.com/data-annotation-tools-market) — automotive/transport = 28.6% of annotation market |
| Language & Localization | [Lara Translate](https://blog.laratranslate.com/ai-translation-trends-2026-to-watch/) — $72B language services | Machine translation market $1.55B (2023), 30%+ CAGR |
| Legal Services | [Sequoia: $60B absorbable](https://sequoiacap.com/article/services-the-new-software/); [Artificial Lawyer](https://www.artificiallawyer.com/2026/03/30/autopilots-can-absorb-60bn-of-legal-work-sequoia/) | Estimated from legal tech reports |
| Business & Digital + Safety | [Sequoia: digital services + cybersecurity](https://sequoiacap.com/article/services-the-new-software/) | Emerging segment |

### Key Investment Signals

**UNDERINVESTED sectors** (large TAM, few annotation jobs — potential opportunities):

- **Engineering & Industrial**: 28.6% of the total annotation tools market by data volume (computer vision, LiDAR, autonomous vehicles), but only 4.1% of the jobs in our dataset. The discrepancy is because much of this annotation is image/video (not LLM text) and dominated by tools like Scale AI Remotasks and Appen. LLM-specific engineering training is an emerging gap.

- **Healthcare & Life Sciences**: 18% of the AI training dataset market with the fastest CAGR (29.7%), but only 7.3% of annotation job openings. Growing — Invisible/Meridial added 5 medical specialty roles, Hippocratic AI ($137M) is hiring RNs at $120-180K base, and Centific has oncology/critical care roles. Still underinvested given the TAM. The US faces a physician shortage of 37,800-124,000 by 2034 ([AAMC](https://www.aamc.org/)), making it harder to find qualified medical annotators.

- **Legal Services**: Sequoia explicitly says "autopilots can absorb $60B of legal work." Legal jumped from 2.9% to 5.3% of the dataset with Harvey AI's 7 new roles, but at $200-300K + equity these are full-time product roles, not annotation gigs. The freelance annotation side remains thin — the J.D. barrier and liability risk means legal training data demands *very* high quality. Harvey's entry suggests legal AI is transitioning from "underinvested" to "heating up."

- **Language & Localization**: $72B language services market with 30%+ CAGR in machine translation, but only 9 annotation openings. Many language roles are embedded in other sectors (Alignerr's 16 language trainers are in Content & Creative).

**SHRINKING sectors** (automation replacing human workers):

- **IT & Data Services**: Basic annotation ($10-12/hr) is being automated by AI-assisted labeling tools (Encord, Roboflow, Snorkel). The 27% market share by volume will shrink as a share of *human* labor spend.

- **General AI Training**: Generalist roles are being replaced by specialists. xAI's Sep 2025 layoff of 500 generalists and pivot to domain specialists is the clearest signal.

---

## Summary Table

| Sector (Sequoia vertical) | Jobs | % | Pay Floor | Pay Ceiling | Typical YOE | Sequoia TAM |
|---------------------------|-----:|---:|----------:|------------:|-------------|-------------|
| Content & Creative Services | 51 | 20.8% | $14/hr | $150/hr | Varies; professional preferred | Part of Media & Content |
| STEM & Research | 32 | 13.1% | $8/hr | $175/hr | Master's–PhD (6-10+ yrs) | R&D Services ($100+B) |
| Education & Training | 28 | 11.4% | $8/hr | $150/hr | Professional (3-5+ yrs) | Education ($150+B) |
| IT & Data Services | 24 | 9.8% | $4/hr | $40/hr | Entry–varies | IT Managed Services ($100+B) |
| Software Engineering | 24 | 9.8% | $8/hr | $150/hr | Professional (3-5+ yrs) | Software Dev ($200+B) |
| Accounting & Finance | 22 | 9.0% | $8/hr | $150/hr | Professional + CPA preferred | Accounting & Audit ($50-80B) |
| General AI Training | 19 | 7.8% | $8/hr | $150/hr | Professional (3-5+ yrs) | N/A (meta-category) |
| Healthcare & Life Sciences | 19 | 7.3% | $8/hr | $180K base | Master's–PhD, RN/MD | Healthcare Rev Cycle ($50-80B) |
| Engineering & Industrial | 10 | 4.1% | $3/hr | $150/hr | Master's (2-5 yrs) | Engineering ($150+B) |
| Language & Localization | 9 | 3.7% | $8/hr | $150/hr | Professional (3-5+ yrs) | Language Services ($72B) |
| Legal Services | 14 (7+7†) | 5.3% | $20/hr | $300K + equity† | J.D. + 3yr top firm (Harvey†); professional (5+ yrs) | Legal ($60B absorbable) |
| Business & Digital Services | 1 | 0.4% | $30/hr | $60/hr | Professional | Digital Services ($50+B) |
| AI Safety & Security | 1 | 0.4% | $15/hr | $50/hr | Professional | Cybersecurity / Risk |

*† = includes Corporate/In-House roles (Harvey AI: 7 legal, Hippocratic AI: 1 healthcare). These are full-time salaried product roles, not annotation/contract work. Annotation-only Legal count is 7; annotation-only Healthcare count is 18.*

---

## Detailed Cluster Analysis

### 1. Content & Creative Services (51 jobs, 20.8%)

**Sequoia mapping:** High-intelligence, increasingly outsourced. The writing/editing/content layer
is the most automatable portion of creative services — and also the portion most needed for RLHF
(preference ranking, style evaluation, creative writing assessment).

| Attribute | Value |
|-----------|-------|
| Pay range | $14–$150/hr |
| Median pay band | ~$40–$65/hr (specialist writers); $14-30/hr (generalist) |
| Typical YOE | Varies widely; 37 unspecified, 14 require professional experience |
| Top employers | Labelbox/Alignerr (25), Surge AI/DataAnnotation (9), Invisible/Meridial (5) |
| Intelligence vs Judgment | Mostly intelligence (grammar, style adherence) with judgment for creative quality |

**Typical roles:** Writing Specialist, Content Review & Evaluation, AI Training - Writing Expert,
Freelance Copywriter, Proofreader, Associate Editor, Creative Writing Trainer, Digital Content Editor

**What's happening:** This is the largest single cluster because every AI lab needs writers to
evaluate model outputs. Labelbox/Alignerr dominates with 25 listings covering 16+ languages and
creative writing. Pay is bimodal — generalist writing at $14-30/hr, specialist (creative, technical) at $40-150/hr.

---

### 2. STEM & Research (32 jobs, 13.1%)

**Sequoia mapping:** High-judgment, insourced-trending. PhD-level expertise can't be commoditized.
These roles train models on scientific reasoning — the hardest problems for LLMs.

| Attribute | Value |
|-----------|-------|
| Pay range | $8–$175/hr |
| Median pay band | $40–$65/hr |
| Typical YOE | 10 require PhD, 2 require Master's, 15 require professional expertise |
| Top employers | Labelbox/Alignerr (9), Invisible/Meridial (7), Surge AI/DataAnnotation (6), Scale AI/Outlier (5) |
| Intelligence vs Judgment | High judgment — evaluating whether AI got the physics/chemistry right |

**Typical roles:** AI Data Science Tutor, Physics Expert, Chemistry Expert, Mathematics Trainer,
Data Analyst Expert, Research Analyst, Biostatistician

**What's happening:** Second largest cluster. The pay ceiling ($175/hr for PhD specialists) is
the highest across all sectors. Outlier AI openly advertises "up to $80/hr" for physics and
medicine experts. This is where Sequoia's "judgment premium" is most visible — you can't
automate evaluating whether an AI's quantum mechanics answer is correct without being a physicist.

---

### 3. Education & Training (28 jobs, 11.4%)

**Sequoia mapping:** Judgment-heavy, traditionally insourced. The "AI tutor" role — teaching
models how to teach — sits at the intersection of education services and AI.

| Attribute | Value |
|-----------|-------|
| Pay range | $8–$150/hr |
| Median pay band | $20–$35/hr (tutors); $60-95/hr W-2 (xAI full-time) |
| Typical YOE | Professional experience required in 24 of 28 roles |
| Top employers | xAI (17), BUKI (5), Surge AI/DataAnnotation (3) |
| Intelligence vs Judgment | Mostly judgment — pedagogical skill, adapting to learner needs |

**Typical roles:** AI Tutor (Full-Time), AI Tutor - Bilingual, AI Tutor Lead, K-12 Education Expert,
ESL Tutor, High School Teacher

**What's happening:** xAI dominates this cluster (17 of 28) after pivoting from generalist annotators
to "specialist AI tutors" in Sep 2025. They cut 500 generalists and are actively expanding specialist
tutors in STEM, finance, medicine, and law. BUKI focuses on multilingual tutoring.

---

### 4. IT & Data Services (24 jobs, 9.8%)

**Sequoia mapping:** IT Managed Services ($100+B TAM). High-intelligence, highly outsourced.
Basic annotation/labeling is the "managed services" layer of the AI stack.

| Attribute | Value |
|-----------|-------|
| Pay range | $4–$40/hr |
| Median pay band | $10–$12/hr |
| Typical YOE | 16 unspecified / entry-friendly; 8 require professional experience |
| Top employers | Surge AI/DataAnnotation (3), Taskify AI (3), OpenTrain AI (2), Innodata (2) |
| Intelligence vs Judgment | Pure intelligence — following annotation guidelines, labeling data |

**Typical roles:** Data Annotator, AI Data Annotation Participant, AI Data Trainer, Search Quality Rater,
Crowdsourced Annotator

**What's happening:** Lowest pay cluster by far ($4-40/hr, median ~$10-12). This is the commoditized
base of the annotation pyramid. Remotasks, TaskUp, and general Appen-style crowdsourcing live here.
Sequoia would classify this as ripe for full automation — and indeed, AI-assisted labeling tools
(Encord, Roboflow) are actively replacing this layer.

---

### 5. Software Engineering (24 jobs, 9.8%)

**Sequoia mapping:** Software Engineering — the article's lead example. Writing code is "intelligence"
(rule-following); knowing *what* to build is "judgment."

| Attribute | Value |
|-----------|-------|
| Pay range | $8–$150/hr |
| Median pay band | $40–$60/hr |
| Typical YOE | 13 require professional experience; 1 Master's, 1 Bachelor's |
| Top employers | Surge AI/DataAnnotation (8), Labelbox/Alignerr (4), NVIDIA (4), Scale AI/Outlier (3) |
| Intelligence vs Judgment | Mixed — evaluating code correctness is intelligence; evaluating code *quality* is judgment |

**Typical roles:** AI Training - Coding Expert, AI Trainer - Coding Specialist, Software Engineer - AI Trainer,
SOQL Developer, API Test Engineer, Creative Developer, ML Engineer

**What's happening:** Coding trainers are among the highest-paid annotation roles because they need
to both write and evaluate code. NVIDIA (via Sustainable Talent) pays $40-95/hr W-2 for annotation
engineers — effectively software engineering compensation. This is the sector where AI is
simultaneously the product being trained *and* the tool that may eventually replace the trainers.

---

### 6. Accounting & Finance (22 jobs, 9.0%)

**Sequoia mapping:** Accounting & Audit ($50-80B TAM outsourced in US). The US has lost ~340,000
accountants over five years while demand has grown; 75% of CPAs nearing retirement.

| Attribute | Value |
|-----------|-------|
| Pay range | $8–$150/hr |
| Median pay band | $30–$60/hr |
| Typical YOE | 18 professional, 3 Master's — CPA strongly preferred |
| Top employers | xAI (17), Scale AI/Outlier (2), Invisible/Meridial (1) |
| Intelligence vs Judgment | Mixed — financial calculations are intelligence; risk assessment is judgment |

**Typical roles:** AI Tutor - Crypto, AI Tutor - Personal Finance, Finance Expert, Accounting Expert,
Actuary, Securities & Commodities Specialist, Financial Analyst

**What's happening:** xAI is the dominant hirer (17 of 22) reflecting their bet on finance tutors
post-pivot. Outlier pays $30-60/hr for accounting and finance experts. The CPA shortage makes this
a sector where AI training is urgent — models need to learn accounting standards from the shrinking
pool of human experts before they retire.

---

### 7. General AI Training (19 jobs, 7.8%)

Catch-all for roles that don't fit a specific Sequoia vertical — general-purpose AI trainers,
evaluators, and ops roles that span multiple domains.

| Attribute | Value |
|-----------|-------|
| Pay range | $8–$150/hr |
| Median pay band | $14–$65/hr |
| Typical YOE | 14 professional, 1 entry-level |
| Top employers | xAI (4), Invisible/Meridial (4), Scale AI (3) |

---

### 8. Healthcare & Life Sciences (19 jobs, 7.3%)

**Sequoia mapping:** Healthcare Revenue Cycle ($50-80B TAM outsourced in US). Clinical knowledge
is high-judgment; billing and coding is high-intelligence.

| Attribute | Value |
|-----------|-------|
| Pay range | $8–$180K base |
| Median pay band | $40–$75/hr |
| Typical YOE | 8 professional (MD/RN/PharmD), 4 Master's, 2 PhD |
| Top employers | Invisible/Meridial (5), Surge AI/DataAnnotation (4), xAI (3), Centific (2), Labelbox/Alignerr (1), Scale AI/Outlier (1), Hippocratic AI (1), Innodata (1) |
| Intelligence vs Judgment | High judgment for clinical reasoning; intelligence for medical coding |

**Typical roles:** Medicine Tutor, Registered Nurse - AI Trainer, Clinical Researcher,
Medical AI Trainer (Internal Medicine, Pharmacology, Pathology, Clinical Diagnostics, Medical Ethics),
Clinical Data Annotation Lead, Senior Oncology Data Abstractor, Critical Care RN

**What's happening:** Healthcare has expanded from 17 to 19 jobs with the addition of specialized
medical AI roles. Invisible Technologies/Meridial now leads with 5 clinical specialty roles
(Internal Medicine, Pharmacology, Pathology, Diagnostics, Ethics) paying $40-75/hr. Hippocratic AI
($137M Series A) is hiring RN/Clinical Product Specialists at $120-180K base — the first full-time,
equity-compensated clinical AI validation role in the dataset. Centific has oncology and critical care
RN annotation roles at $30-55/hr. This mirrors the broader healthcare labor shortage that Sequoia highlights.

**New entrants:**
- **Hippocratic AI** — safety-focused healthcare LLM startup ($137M raised), hiring RNs for AI validation
- **Invisible/Meridial** — 5 medical specialty training roles (MD/PharmD-level)
- **Centific** — oncology data abstraction and critical care RN annotation

---

### 9. Engineering & Industrial (10 jobs, 4.1%)

**Sequoia mapping:** Engineering services — a judgment-heavy, traditionally insourced sector
now being partially outsourced for AI training.

| Attribute | Value |
|-----------|-------|
| Pay range | $3–$150/hr |
| Median pay band | $30–$60/hr |
| Typical YOE | 4 Master's, 4 professional |
| Top employers | Scale AI/Outlier (6), xAI (1), Comrise (1) |

**Typical roles:** Civil/Electrical/Mechanical/Chemical Engineering Expert, Computer Vision Annotator,
Robotics Data Specialist, LiDAR Annotation

---

### 10. Language & Localization (9 jobs, 3.7%)

**Sequoia mapping:** Translation/BPO — high-intelligence, highly outsourced, one of the first
sectors disrupted by AI. The remaining human roles focus on judgment (cultural nuance, idiom quality).

| Attribute | Value |
|-----------|-------|
| Pay range | $8–$150/hr |
| Median pay band | ~$20–$40/hr |
| Typical YOE | 6 professional, 3 varies |
| Top employers | Distributed across LXT, Anuttacon, Welocalize, Gloz, TELUS Digital |

**Note:** Many language-specific roles are classified under Content & Creative (Alignerr's 16 language
trainer roles) or Education (BUKI multilingual tutors). This cluster captures the pure
localization/translation/multilingual QA roles.

---

### 11. Legal Services (14 jobs, 5.3%)

**Sequoia mapping:** Legal Transactional ($20-25B TAM). Contract review, NDA drafting, and
regulatory filings are intelligence-heavy and routinely outsourced. Sequoia estimates AI
"autopilots can absorb $60B of legal work."

| Attribute | Value |
|-----------|-------|
| Pay range | $20/hr – $300K base + equity |
| Median pay band | $45-75/hr (annotation); $200-300K (Harvey full-time) |
| Typical YOE | J.D. + 3yr top-tier firm (Harvey); J.D. or equivalent (others) |
| Top employers | **Harvey AI (7)**, xAI (3), Surge AI/DataAnnotation (2), YO IT Consulting (1), Scale AI/Outlier (1) |
| Intelligence vs Judgment | Contract review = intelligence; strategy/advocacy = judgment |

**Typical roles:** Legal Engineer (Harvey), Legal Engineer - Product Specialist Tax/Innovation,
Applied Legal Researcher, AI Legal and Compliance Tutor, Corporate Law and Securities Expert,
Law Expert, Personal Injury Paralegal

**What's happening:** Legal has **doubled from 7 to 14 jobs** — the fastest-growing sector in our dataset.
Harvey AI is the catalyst: they've posted 7 Legal Engineer roles requiring JD + 3yr at top-tier firms,
paying $200-300K base + equity. This is the highest compensation in the entire dataset and represents
a qualitative shift — from hourly annotation work to full-time, equity-compensated legal AI development.

Harvey's roles span geographic (SF, NYC, Dallas, London/EMEA) and practice area (Tax, Innovation,
Custom Solutions) dimensions, suggesting they're building out a full legal AI training org rather than
ad-hoc annotation. Their Applied Legal Researcher role bridges legal domain expertise with AI research.

The freelance/annotation side remains active too — Outlier pays up to $75/hr for law experts, and
xAI has 3 legal tutor roles.

**New entrant:**
- **Harvey AI** — Legal AI startup (valued at $1.5B+, backed by Sequoia), hiring 7 legal engineers at $200-300K + equity. Requires JD + top-firm experience. The highest-paid AI training roles in the dataset by a wide margin.

---

### 12–13. Business & Digital Services / AI Safety & Security (2 jobs, 0.8%)

Emerging categories with only 1 job each. B2B/B2C digital domain expertise and AI safety
red-teaming. Expect these to grow as models get deployed into more vertical business contexts
and safety requirements intensify.

---

## Key Takeaways

### The Pay Gradient Follows Sequoia's Judgment Spectrum

The data confirms Sequoia's core thesis. Jobs requiring **more judgment** (STEM, Healthcare,
Legal, Software Engineering) pay 3–8x more than jobs requiring **pure intelligence** (IT & Data
Services, basic annotation). The judgment premium is real:

| Intelligence ←→ Judgment | Median Pay | Example |
|--------------------------|-----------|---------|
| Pure intelligence | $10–12/hr | Data annotator, image labeler |
| Mixed | $30–60/hr | Finance expert, coding evaluator |
| High judgment | $60–150/hr | PhD physicist, practicing attorney |

### The Biggest Sectors Are Content and STEM

Over a third of all roles (34%) fall into Content & Creative or STEM & Research. This makes
sense: LLMs are fundamentally language models that need writing quality feedback, and their
biggest weakness is factual/scientific reasoning that needs domain expert correction.

### xAI's Bet on Education

xAI accounts for 48 of 262 jobs (18%) and dominates Education (17/28), Accounting/Finance (17/22),
and Healthcare (3/19). Their Sep 2025 pivot from generalist annotators to specialist "AI tutors"
is a bet that the judgment layer matters more than volume. Harvey AI's entry with 7 legal roles
at $200-300K + equity sets a new pay ceiling for the entire dataset.

### The Commoditized Base Is Shrinking

IT & Data Services (basic annotation) has the lowest pay and is the most vulnerable to automation.
Companies like Encord, Roboflow, and Snorkel are already automating this layer. The future of
human-in-the-loop AI training is domain expertise, not volume annotation.

---

## Methodology

- Jobs classified programmatically using title, tags, description, and requirements keywords
- Pay ranges extracted from compensation fields (178 of 262 jobs have compensation data)
- YOE inferred from degree requirements and experience language in job descriptions
- Sequoia vertical mapping based on the "Services: The New Software" priority map (Mar 2026)
- Some roles span multiple sectors; classified by primary domain signal

---

## References

- [Sequoia Capital — "Services: The New Software"](https://sequoiacap.com/article/services-the-new-software/) (Mar 2026)
- [xAI lays off 500 data annotators, pivots to specialist AI tutors](https://techcrunch.com/2025/09/13/xai-reportedly-lays-off-500-workers-from-data-annotation-team/) (Sep 2025)
- [Surge AI hits $1.2B revenue](https://getlatka.com/blog/surgeai-revenue-valuation/) (2024)
- [Meta acquires 49% of Scale AI for $14.3B](https://www.cnbc.com/2025/06/12/scale-ai-founder-wang-announces-exit-for-meta-part-of-14-billion-deal.html) (Jun 2025)
- [Google terminates $82.8M Appen contract](https://gun.io/news/2025/12/scale-ai-alternatives-for-enterprise-ai-teams/) (Mar 2024)
- [Harvey AI Careers](https://jobs.ashbyhq.com/harvey) — Legal AI startup ($1.5B+ valuation, Sequoia-backed)
- [Hippocratic AI — Safety-focused healthcare LLM](https://www.hippocraticai.com/) — $137M Series A

---
*Generated from 262 job postings in job_data.py. Last updated: 2026-04-11*
