#!/usr/bin/env python3
# Copyright 2023 Hanchung Lee
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comprehensive AI Trainer and Data Annotation Job Postings Database
===================================================================
Contains a curated collection of 169 job postings for AI training,
data annotation, tutoring, and related roles from leading companies.
"""

from scrape_ai_trainer_jobs import JobPosting


def get_all_jobs() -> list[JobPosting]:
    """Return a list of all 169 AI trainer and data annotation job postings."""
    return [
        # xAI Roles (1-47)
        JobPosting(
            title="AI Tutor (Full-Time)",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4595198007",
            source="xAI Careers (Greenhouse)",
            compensation="$35-65/hr",
            job_type="Contract (6-month)",
            description="Teach Grok about human interactions through AI tutoring. Contribute to AI tutor training via annotations and evaluations.",
            requirements=[],
            responsibilities=[
                "Annotate and evaluate interactions",
                "Teach Grok about human behavior",
                "Provide feedback on AI responses"
            ],
            tags=["AI tutor", "Grok", "xAI", "RLHF"]
        ),
        JobPosting(
            title="AI Tutor - Bilingual (Full-Time)",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4512212007",
            source="xAI Careers (Greenhouse)",
            compensation="$35-65/hr",
            job_type="Full-Time",
            description="AI Tutoring role requiring bilingual fluency. Requires English plus one of: Korean, Vietnamese, Chinese, German, Russian, Italian, French, Arabic, Indonesian, Turkish, Hindi, Persian, Spanish, Portuguese.",
            requirements=[
                "English fluency",
                "Fluency in one of: Korean, Vietnamese, Chinese, German, Russian, Italian, French, Arabic, Indonesian, Turkish, Hindi, Persian, Spanish, Portuguese"
            ],
            responsibilities=[
                "Provide AI tutor training in multiple languages",
                "Annotate multilingual interactions",
                "Evaluate AI responses across languages"
            ],
            tags=["Bilingual", "Multilingual", "AI tutor", "xAI", "RLHF"]
        ),
        JobPosting(
            title="AI Tutor Lead",
            company="xAI",
            location="Remote",
            url="https://boards.greenhouse.io/xai/jobs/4305834007",
            source="xAI Careers (Greenhouse)",
            description="Leadership role guiding and mentoring a large team of AI tutors while maintaining data integrity and quality standards.",
            requirements=[
                "Experience leading teams",
                "AI tutor expertise",
                "Data quality management"
            ],
            responsibilities=[
                "Guide and mentor AI tutor team",
                "Maintain data integrity",
                "Oversee team quality standards",
                "Manage tutor training and feedback"
            ],
            tags=["Leadership", "AI tutor", "Team management", "xAI"]
        ),
        JobPosting(
            title="STEM Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4538773007",
            source="xAI Careers (Greenhouse)",
            compensation="$35-65/hr",
            job_type="Full-Time",
            description="STEM expert needed to teach Grok about science, technology, engineering, and mathematics concepts. Requires IMO medalist or Master's/PhD in STEM field.",
            requirements=[
                "IMO medalist status OR Master's degree or PhD in STEM field",
                "Deep STEM expertise"
            ],
            responsibilities=[
                "Teach STEM concepts to AI",
                "Evaluate STEM-related responses",
                "Provide expert feedback on scientific accuracy"
            ],
            tags=["STEM", "Physics", "Math", "Biology", "Chemistry", "Engineering", "xAI"]
        ),
        JobPosting(
            title="AI Data Science Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4621682007",
            source="xAI Careers (Greenhouse)",
            description="Data science expert to train Grok on data science concepts, analysis, and machine learning. Requires IMO medalist or Master's/PhD in data science.",
            requirements=[
                "IMO medalist status OR Master's degree or PhD in Data Science",
                "Advanced data science expertise"
            ],
            responsibilities=[
                "Teach data science concepts",
                "Evaluate data science responses",
                "Provide expert analysis feedback"
            ],
            tags=["Data Science", "Machine Learning", "Analytics", "xAI"]
        ),
        JobPosting(
            title="AI Tutor - Crypto",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5040344007",
            source="xAI Careers (Greenhouse)",
            description="Cryptocurrency and digital asset markets expert. Create annotations and evaluations to train Grok on crypto concepts.",
            requirements=[
                "Cryptocurrency expertise",
                "Digital assets knowledge"
            ],
            responsibilities=[
                "Annotate crypto market interactions",
                "Evaluate crypto-related responses",
                "Provide feedback on blockchain concepts"
            ],
            tags=["Cryptocurrency", "Digital Assets", "Blockchain", "Finance", "xAI"]
        ),
        JobPosting(
            title="Video Games Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4879839007",
            source="xAI Careers (Greenhouse)",
            description="Video game expert to train Grok on game concepts, mechanics, generation, and gameplay. Create annotations and evaluate AI responses.",
            requirements=[
                "Extensive video game knowledge",
                "Gaming expertise"
            ],
            responsibilities=[
                "Teach game concepts and mechanics",
                "Evaluate game-related responses",
                "Train on game generation concepts"
            ],
            tags=["Video Games", "Gaming", "Game Mechanics", "xAI"]
        ),
        JobPosting(
            title="Multilingual Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4879853007",
            source="xAI Careers (Greenhouse)",
            description="Multilingual audio specialist to train Grok on multilingual audio capabilities, voice interactions, and speech recognition.",
            requirements=[
                "Multilingual proficiency",
                "Audio/speech knowledge"
            ],
            responsibilities=[
                "Train on multilingual audio",
                "Evaluate speech recognition",
                "Provide voice interaction feedback"
            ],
            tags=["Multilingual", "Audio", "Speech Recognition", "Voice", "xAI"]
        ),
        JobPosting(
            title="Image Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5047544007",
            source="xAI Careers (Greenhouse)",
            description="Visual content specialist to train Grok on interpreting and generating visual content, images, and visual understanding.",
            requirements=[
                "Visual/image expertise",
                "Computer vision knowledge"
            ],
            responsibilities=[
                "Train on image interpretation",
                "Evaluate image generation",
                "Provide visual content feedback"
            ],
            tags=["Computer Vision", "Image Generation", "Visual Content", "xAI"]
        ),
        JobPosting(
            title="Image & Video Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4879781007",
            source="xAI Careers (Greenhouse)",
            description="Cinematography and videography expert to train Grok on visual media production, composition, and techniques.",
            requirements=[
                "Cinematography expertise",
                "Videography experience",
                "Visual production knowledge"
            ],
            responsibilities=[
                "Train on cinematography concepts",
                "Evaluate video content",
                "Provide production feedback"
            ],
            tags=["Cinematography", "Videography", "Film", "Visual Production", "xAI"]
        ),
        JobPosting(
            title="Video Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5047564007",
            source="xAI Careers (Greenhouse)",
            description="Video content specialist to train Grok on interpreting and generating video content.",
            requirements=[
                "Video expertise",
                "Video analysis knowledge"
            ],
            responsibilities=[
                "Train on video interpretation",
                "Evaluate video generation",
                "Provide video content feedback"
            ],
            tags=["Video Generation", "Video Interpretation", "Video Content", "xAI"]
        ),
        JobPosting(
            title="3D Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5045788007",
            source="xAI Careers (Greenhouse)",
            description="3D content generation specialist to train Grok on 3D modeling, visualization, and spatial concepts.",
            requirements=[
                "3D modeling expertise",
                "3D graphics knowledge"
            ],
            responsibilities=[
                "Train on 3D concepts",
                "Evaluate 3D generation",
                "Provide 3D content feedback"
            ],
            tags=["3D Modeling", "3D Generation", "Graphics", "xAI"]
        ),
        JobPosting(
            title="Audio Editing Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4879783007",
            source="xAI Careers (Greenhouse)",
            description="Audio expert to train Grok on audio editing, voice interactions, and sound processing.",
            requirements=[
                "Audio editing expertise",
                "Sound processing knowledge"
            ],
            responsibilities=[
                "Train on audio editing concepts",
                "Evaluate audio processing",
                "Provide sound feedback"
            ],
            tags=["Audio Editing", "Sound Processing", "Voice", "xAI"]
        ),
        JobPosting(
            title="AI Tutor - Audio Editing",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5098930007",
            source="xAI Careers (Greenhouse)",
            description="Audio editing specialist to annotate and evaluate audio-related content for AI training.",
            requirements=[
                "Professional audio editing experience",
                "Sound design knowledge"
            ],
            responsibilities=[
                "Annotate audio content",
                "Evaluate audio quality",
                "Provide editing feedback"
            ],
            tags=["Audio Editing", "Sound Design", "Professional Audio", "xAI"]
        ),
        JobPosting(
            title="Personality & Behavior Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4879779007",
            source="xAI Careers (Greenhouse)",
            description="Tutor to train Grok's personality and behavior. Requires expertise in Communications, Creative Writing, or Performing Arts.",
            requirements=[
                "Communications expertise OR",
                "Creative Writing background OR",
                "Performing Arts experience"
            ],
            responsibilities=[
                "Train on personality traits",
                "Guide behavioral responses",
                "Evaluate personality consistency"
            ],
            tags=["Personality", "Behavior", "Communications", "Creative Writing", "xAI"]
        ),
        JobPosting(
            title="Writing Specialist",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5017529007",
            source="xAI Careers (Greenhouse)",
            description="Writing expert to improve Grok's writing outputs, style, clarity, and quality.",
            requirements=[
                "Professional writing expertise",
                "Writing evaluation skills"
            ],
            responsibilities=[
                "Evaluate writing quality",
                "Provide writing feedback",
                "Train on writing best practices"
            ],
            tags=["Writing", "Content", "Writing Quality", "xAI"]
        ),
        JobPosting(
            title="Medicine Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4855858007",
            source="xAI Careers (Greenhouse)",
            description="Medical domain expert to train Grok on medical concepts, procedures, and healthcare knowledge.",
            requirements=[
                "Medical domain expertise",
                "Healthcare background"
            ],
            responsibilities=[
                "Train on medical concepts",
                "Evaluate medical accuracy",
                "Provide healthcare feedback"
            ],
            tags=["Medicine", "Healthcare", "Medical", "xAI"]
        ),
        JobPosting(
            title="AI Healthcare and Administration Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4931312007",
            source="xAI Careers (Greenhouse)",
            description="Healthcare and administrative specialist to train Grok on healthcare systems, administration, and operations.",
            requirements=[
                "Healthcare expertise",
                "Administrative knowledge"
            ],
            responsibilities=[
                "Train on healthcare systems",
                "Evaluate healthcare responses",
                "Provide administrative feedback"
            ],
            tags=["Healthcare", "Administration", "Healthcare Systems", "xAI"]
        ),
        JobPosting(
            title="AI Tutor - English (Foreign Accents)",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5098929007",
            source="xAI Careers (Greenhouse)",
            description="English language specialist with expertise in non-native accents. Train Grok on understanding accented English.",
            requirements=[
                "English fluency",
                "Foreign accent expertise"
            ],
            responsibilities=[
                "Train on accented English",
                "Evaluate accent comprehension",
                "Provide speech feedback"
            ],
            tags=["English", "Accents", "Speech", "Linguistics", "xAI"]
        ),
        JobPosting(
            title="AI Tutor - Polish",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5090218007",
            source="xAI Careers (Greenhouse)",
            description="Polish language specialist to train Grok on Polish language, culture, and communication.",
            requirements=[
                "Polish fluency",
                "Native or near-native proficiency"
            ],
            responsibilities=[
                "Train on Polish language",
                "Evaluate Polish comprehension",
                "Provide language feedback"
            ],
            tags=["Polish", "Language", "Multilingual", "xAI"]
        ),
        JobPosting(
            title="AI Tutor - Danish",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5090189007",
            source="xAI Careers (Greenhouse)",
            description="Danish language specialist to train Grok on Danish language and culture.",
            requirements=[
                "Danish fluency",
                "Native or near-native proficiency"
            ],
            responsibilities=[
                "Train on Danish language",
                "Evaluate Danish comprehension",
                "Provide language feedback"
            ],
            tags=["Danish", "Language", "Multilingual", "xAI"]
        ),
        JobPosting(
            title="AI Tutor - Finnish",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5090199007",
            source="xAI Careers (Greenhouse)",
            description="Finnish language specialist to train Grok on Finnish language and culture.",
            requirements=[
                "Finnish fluency",
                "Native or near-native proficiency"
            ],
            responsibilities=[
                "Train on Finnish language",
                "Evaluate Finnish comprehension",
                "Provide language feedback"
            ],
            tags=["Finnish", "Language", "Multilingual", "xAI"]
        ),
        JobPosting(
            title="AI Tutor - Italian",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5090209007",
            source="xAI Careers (Greenhouse)",
            description="Italian language specialist to train Grok on Italian language and culture.",
            requirements=[
                "Italian fluency",
                "Native or near-native proficiency"
            ],
            responsibilities=[
                "Train on Italian language",
                "Evaluate Italian comprehension",
                "Provide language feedback"
            ],
            tags=["Italian", "Language", "Multilingual", "xAI"]
        ),
        JobPosting(
            title="AI Tutor - Thai",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5090272007",
            source="xAI Careers (Greenhouse)",
            description="Thai language specialist to train Grok on Thai language and culture.",
            requirements=[
                "Thai fluency",
                "Native or near-native proficiency"
            ],
            responsibilities=[
                "Train on Thai language",
                "Evaluate Thai comprehension",
                "Provide language feedback"
            ],
            tags=["Thai", "Language", "Multilingual", "xAI"]
        ),
        JobPosting(
            title="AI Legal and Compliance Tutor",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4931315007",
            source="xAI Careers (Greenhouse)",
            compensation="$45-100/hr",
            description="Legal expert with J.D. to train Grok on legal concepts, compliance, and regulatory knowledge.",
            requirements=[
                "Juris Doctor (J.D.) degree required",
                "Legal expertise"
            ],
            responsibilities=[
                "Train on legal concepts",
                "Evaluate legal accuracy",
                "Provide compliance feedback"
            ],
            tags=["Legal", "Compliance", "Law", "Regulatory", "xAI"]
        ),
        JobPosting(
            title="AI Tutor - Legal Specialist",
            company="xAI",
            location="Remote",
            url="https://boards.greenhouse.io/xai/jobs/4615618007",
            source="xAI Careers (Greenhouse)",
            description="Legal specialization role to annotate and evaluate legal content for AI training.",
            requirements=[
                "Legal background",
                "Legal writing expertise"
            ],
            responsibilities=[
                "Annotate legal content",
                "Evaluate legal quality",
                "Provide legal feedback"
            ],
            tags=["Legal", "Law", "Legal Specialization", "xAI"]
        ),
        JobPosting(
            title="Corporate Law and Securities Expert",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5040428007",
            source="xAI Careers (Greenhouse)",
            description="Corporate law and securities expert to train Grok on securities filings, M&A documents, and corporate governance.",
            requirements=[
                "Corporate law expertise",
                "Securities law knowledge",
                "M&A experience"
            ],
            responsibilities=[
                "Train on securities concepts",
                "Evaluate corporate documents",
                "Provide governance feedback"
            ],
            tags=["Corporate Law", "Securities", "M&A", "Governance", "xAI"]
        ),
        JobPosting(
            title="AI Tutor - Personal Finance",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4933195007",
            source="xAI Careers (Greenhouse)",
            description="Personal finance expert to annotate and evaluate personal finance content for AI training.",
            requirements=[
                "Personal finance expertise",
                "Financial planning knowledge"
            ],
            responsibilities=[
                "Annotate finance content",
                "Evaluate financial accuracy",
                "Provide finance feedback"
            ],
            tags=["Personal Finance", "Finance", "Financial Planning", "xAI"]
        ),
        JobPosting(
            title="Finance Expert",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4760806007",
            source="xAI Careers (Greenhouse)",
            description="General finance expert covering equities, commodities, real estate, fixed income, forex, and derivatives.",
            requirements=[
                "Broad finance expertise",
                "Multiple asset class knowledge"
            ],
            responsibilities=[
                "Train on finance concepts",
                "Evaluate financial responses",
                "Provide finance expertise"
            ],
            tags=["Finance", "Equities", "Commodities", "Real Estate", "Fixed Income", "Forex", "Derivatives", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Corporate Finance",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5040407007",
            source="xAI Careers (Greenhouse)",
            description="Corporate finance specialist focusing on strategic planning and capital allocation.",
            requirements=[
                "Corporate finance expertise",
                "Strategic planning knowledge"
            ],
            responsibilities=[
                "Train on corporate finance",
                "Evaluate capital allocation",
                "Provide strategic feedback"
            ],
            tags=["Corporate Finance", "Strategic Planning", "Capital Allocation", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Equity Research",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4933173007",
            source="xAI Careers (Greenhouse)",
            description="Equity research specialist covering capital markets, fundamental analysis, and valuation.",
            requirements=[
                "Equity research expertise",
                "Capital markets knowledge",
                "Valuation experience"
            ],
            responsibilities=[
                "Train on equity research",
                "Evaluate analysis quality",
                "Provide valuation feedback"
            ],
            tags=["Equity Research", "Capital Markets", "Valuation", "Fundamental Analysis", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Fixed Income",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5039375007",
            source="xAI Careers (Greenhouse)",
            description="Fixed income specialist covering bond trading and interest rate derivatives.",
            requirements=[
                "Fixed income expertise",
                "Bond trading knowledge",
                "Derivatives experience"
            ],
            responsibilities=[
                "Train on fixed income",
                "Evaluate bond analysis",
                "Provide derivatives feedback"
            ],
            tags=["Fixed Income", "Bonds", "Derivatives", "Interest Rate", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Quant",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4922806007",
            source="xAI Careers (Greenhouse)",
            description="Quantitative finance specialist focusing on algorithmic investment strategies.",
            requirements=[
                "Quantitative finance expertise",
                "Algorithmic trading knowledge",
                "Math/statistics background"
            ],
            responsibilities=[
                "Train on quant concepts",
                "Evaluate algorithms",
                "Provide quantitative feedback"
            ],
            tags=["Quantitative Finance", "Algorithms", "Trading Strategies", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Quantitative Trading",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5040333007",
            source="xAI Careers (Greenhouse)",
            description="Quantitative trading specialist to train Grok on algorithmic and quantitative trading strategies.",
            requirements=[
                "Quantitative trading expertise",
                "Trading systems knowledge"
            ],
            responsibilities=[
                "Train on trading strategies",
                "Evaluate trading concepts",
                "Provide quant feedback"
            ],
            tags=["Quantitative Trading", "Trading Strategies", "Algorithmic Trading", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Structured Finance",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5040389007",
            source="xAI Careers (Greenhouse)",
            description="Structured finance expert covering ABS, CMBS, MBS, and CLOs.",
            requirements=[
                "Structured finance expertise",
                "Securitization knowledge",
                "Asset-backed securities experience"
            ],
            responsibilities=[
                "Train on structured products",
                "Evaluate securitization",
                "Provide structured finance feedback"
            ],
            tags=["Structured Finance", "ABS", "CMBS", "MBS", "CLOs", "Securitization", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Credit Analyst",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5039377007",
            source="xAI Careers (Greenhouse)",
            description="Credit analyst specialist focusing on private credit analysis and evaluation.",
            requirements=[
                "Credit analysis expertise",
                "Private credit knowledge"
            ],
            responsibilities=[
                "Train on credit analysis",
                "Evaluate credit quality",
                "Provide credit feedback"
            ],
            tags=["Credit Analysis", "Private Credit", "Risk Analysis", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Private Credit",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5039376007",
            source="xAI Careers (Greenhouse)",
            description="Private credit investment specialist to train Grok on private credit markets and investing.",
            requirements=[
                "Private credit expertise",
                "Credit investing knowledge"
            ],
            responsibilities=[
                "Train on private credit",
                "Evaluate investments",
                "Provide credit feedback"
            ],
            tags=["Private Credit", "Credit Investing", "Investment", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Portfolio Management",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4933179007",
            source="xAI Careers (Greenhouse)",
            description="Portfolio management specialist covering portfolio construction and risk management.",
            requirements=[
                "Portfolio management expertise",
                "Risk management knowledge"
            ],
            responsibilities=[
                "Train on portfolio concepts",
                "Evaluate construction",
                "Provide risk feedback"
            ],
            tags=["Portfolio Management", "Risk Management", "Asset Allocation", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Risk",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5040365007",
            source="xAI Careers (Greenhouse)",
            description="Risk management specialist focusing on risk quantification and stress testing.",
            requirements=[
                "Risk management expertise",
                "Quantitative risk knowledge",
                "Stress testing experience"
            ],
            responsibilities=[
                "Train on risk concepts",
                "Evaluate risk models",
                "Provide risk feedback"
            ],
            tags=["Risk Management", "Risk Quantification", "Stress Testing", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - FICC Research",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5039381007",
            source="xAI Careers (Greenhouse)",
            description="FICC (Fixed Income, Currencies, Commodities) research specialist.",
            requirements=[
                "FICC expertise",
                "Fixed income knowledge",
                "FX and commodities experience"
            ],
            responsibilities=[
                "Train on FICC concepts",
                "Evaluate market analysis",
                "Provide research feedback"
            ],
            tags=["FICC", "Fixed Income", "Currencies", "Commodities", "Research", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Macro Research Analyst",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5039410007",
            source="xAI Careers (Greenhouse)",
            description="Macroeconomic research analyst covering global macro analysis and forecasting.",
            requirements=[
                "Macro analysis expertise",
                "Economics knowledge",
                "Global markets understanding"
            ],
            responsibilities=[
                "Train on macro concepts",
                "Evaluate economic analysis",
                "Provide macro feedback"
            ],
            tags=["Macro Research", "Economics", "Global Markets", "Analysis", "xAI"]
        ),
        JobPosting(
            title="Finance Expert - Real Estate Investment",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5040379007",
            source="xAI Careers (Greenhouse)",
            description="Real estate investment specialist to train Grok on real estate markets and investing.",
            requirements=[
                "Real estate expertise",
                "Investment knowledge",
                "Property market understanding"
            ],
            responsibilities=[
                "Train on real estate",
                "Evaluate investments",
                "Provide RE feedback"
            ],
            tags=["Real Estate", "Real Estate Investment", "Property Markets", "xAI"]
        ),
        JobPosting(
            title="Investment Banking Expert - M&A",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4933187007",
            source="xAI Careers (Greenhouse)",
            description="Investment banking specialist focusing on M&A transactions and advisory.",
            requirements=[
                "Investment banking expertise",
                "M&A experience"
            ],
            responsibilities=[
                "Train on M&A concepts",
                "Evaluate transactions",
                "Provide IB feedback"
            ],
            tags=["Investment Banking", "M&A", "Corporate Advisory", "xAI"]
        ),
        JobPosting(
            title="Investment Banking Expert - DCM",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/5039373007",
            source="xAI Careers (Greenhouse)",
            description="Debt capital markets specialist in investment banking.",
            requirements=[
                "DCM expertise",
                "Debt markets knowledge"
            ],
            responsibilities=[
                "Train on DCM concepts",
                "Evaluate debt offerings",
                "Provide DCM feedback"
            ],
            tags=["Debt Capital Markets", "DCM", "Investment Banking", "xAI"]
        ),
        JobPosting(
            title="Investment Banking Expert - ECM",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4954568007",
            source="xAI Careers (Greenhouse)",
            description="Equity capital markets specialist in investment banking.",
            requirements=[
                "ECM expertise",
                "Equity markets knowledge"
            ],
            responsibilities=[
                "Train on ECM concepts",
                "Evaluate equity offerings",
                "Provide ECM feedback"
            ],
            tags=["Equity Capital Markets", "ECM", "Investment Banking", "xAI"]
        ),
        JobPosting(
            title="Economics Expert",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4922802007",
            source="xAI Careers (Greenhouse)",
            description="Economics expert covering macroeconomic forecasting and microeconomic incentives.",
            requirements=[
                "Economics expertise",
                "Macro/micro knowledge"
            ],
            responsibilities=[
                "Train on economics",
                "Evaluate economic analysis",
                "Provide economics feedback"
            ],
            tags=["Economics", "Macroeconomics", "Microeconomics", "Forecasting", "xAI"]
        ),
        JobPosting(
            title="Accounting Expert",
            company="xAI",
            location="Remote",
            url="https://job-boards.greenhouse.io/xai/jobs/4922800007",
            source="xAI Careers (Greenhouse)",
            description="Accounting expert with Big Four experience covering financial reporting and GAAP.",
            requirements=[
                "Accounting expertise",
                "Big Four or equivalent experience",
                "GAAP knowledge"
            ],
            responsibilities=[
                "Train on accounting",
                "Evaluate financial reporting",
                "Provide accounting feedback"
            ],
            tags=["Accounting", "GAAP", "Financial Reporting", "Big Four", "xAI"]
        ),
        JobPosting(
            title="Data Annotator",
            company="xAI",
            location="Remote",
            url="https://jobs.weekday.works/xai-data-annotator",
            source="xAI Careers (Weekday)",
            description="Data annotator role for xAI training data.",
            requirements=[],
            responsibilities=[
                "Annotate training data",
                "Label datasets",
                "Ensure data quality"
            ],
            tags=["Data Annotation", "Labeling", "xAI"]
        ),
        # SuperAnnotate (49-50)
        JobPosting(
            title="AI Data Trainer",
            company="SuperAnnotate",
            location="Remote",
            url="https://www.superannotate.com/careers",
            source="SuperAnnotate Careers",
            compensation="Project-based pay",
            job_type="Freelance",
            description="Create prompts, evaluate and rank AI responses, test for biases in machine learning models. Freelance project-based work.",
            requirements=[
                "AI/ML knowledge",
                "Prompt creation skills",
                "Bias evaluation expertise"
            ],
            responsibilities=[
                "Create evaluation prompts",
                "Rank AI responses",
                "Test for biases",
                "Provide feedback"
            ],
            tags=["AI Training", "Freelance", "Prompt Engineering", "Bias Testing", "SuperAnnotate"]
        ),
        JobPosting(
            title="Subject Matter Expert - AI Training",
            company="SuperAnnotate",
            location="Remote",
            url="https://sme.careers/",
            source="SME Careers (SuperAnnotate)",
            description="Subject matter expert contributing to AI training initiatives through SuperAnnotate.",
            requirements=[
                "Domain expertise",
                "AI training knowledge"
            ],
            responsibilities=[
                "Provide domain expertise",
                "Train AI models",
                "Evaluate responses"
            ],
            tags=["Subject Matter Expert", "SME", "Domain Expertise", "SuperAnnotate"]
        ),
        # YO IT Consulting (51-53)
        JobPosting(
            title="AI Trainer/Data Annotator - Remote",
            company="YO IT Consulting",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=03e62b9e11b929ba",
            source="Indeed",
            job_type="Part-Time",
            compensation="Weekly pay via Stripe/Wise",
            description="AI trainer and data annotator role. Approximately 20 hours per week with weekly payments via Stripe or Wise.",
            requirements=[
                "Data annotation experience",
                "AI/ML knowledge"
            ],
            responsibilities=[
                "Annotate training data",
                "Train AI models",
                "Quality assurance"
            ],
            tags=["AI Training", "Data Annotation", "Part-Time", "Remote", "YO IT"]
        ),
        JobPosting(
            title="AI Trainer - Remote",
            company="YO IT Consulting",
            location="Remote",
            url="https://www.glassdoor.com/job-listing/ai-trainer-remote-yo-it-consulting-JV_KO0,17_KE18,34.htm?jl=1010044412091",
            source="Glassdoor",
            description="Remote AI trainer position with YO IT Consulting.",
            requirements=[
                "AI training experience",
                "Remote work capability"
            ],
            responsibilities=[
                "Train AI models",
                "Annotate data",
                "Provide feedback"
            ],
            tags=["AI Training", "Remote", "YO IT"]
        ),
        JobPosting(
            title="Legal Expert - AI Trainer",
            company="YO IT Consulting",
            location="Remote",
            url="https://www.usaremotejobs.app/job/yo-it-consulting-legal-expert-ai-trainer",
            source="USA Remote Jobs",
            description="Legal expert position to train AI models with legal knowledge and expertise.",
            requirements=[
                "Legal background",
                "AI training knowledge"
            ],
            responsibilities=[
                "Train on legal concepts",
                "Annotate legal content",
                "Provide legal expertise"
            ],
            tags=["Legal", "AI Training", "Legal Expert", "YO IT"]
        ),
        # Prolific (54)
        JobPosting(
            title="AI Data Annotation Participant",
            company="Prolific",
            location="Remote",
            url="https://www.prolific.com/data-annotation",
            source="Prolific",
            compensation="$8/£6 per hour",
            job_type="Freelance",
            description="Participate in AI data annotation studies through the Prolific platform. Study-based compensation ranging from $8-6 per hour.",
            requirements=[
                "English proficiency",
                "Ability to focus on detail"
            ],
            responsibilities=[
                "Annotate data",
                "Complete studies",
                "Provide feedback"
            ],
            tags=["Data Annotation", "Crowdsourcing", "Freelance", "Prolific"]
        ),
        # HumanSignal (55-56)
        JobPosting(
            title="Data Annotator - Architectural Floor Plans",
            company="HumanSignal",
            location="Remote",
            url="https://humansignal.com/careers/",
            source="HumanSignal Careers",
            description="Label architectural floor plans using Label Studio Enterprise for AI training.",
            requirements=[
                "Architectural knowledge",
                "Attention to detail"
            ],
            responsibilities=[
                "Label floor plans",
                "Use Label Studio",
                "Ensure annotation quality"
            ],
            tags=["Data Annotation", "Architecture", "Label Studio", "HumanSignal"]
        ),
        JobPosting(
            title="Content Review & Evaluation Specialist",
            company="HumanSignal",
            location="Remote",
            url="https://job-boards.greenhouse.io/humansignal",
            source="HumanSignal Careers",
            description="Review and evaluate content for AI model training and quality assurance.",
            requirements=[
                "Content evaluation skills",
                "Critical thinking"
            ],
            responsibilities=[
                "Review content",
                "Evaluate quality",
                "Provide feedback"
            ],
            tags=["Content Review", "Evaluation", "Quality Assurance", "HumanSignal"]
        ),
        # DataAnnotation.tech (57-62)
        # NOTE: DataAnnotation.tech uses a single landing page for all roles,
        # so we append #<track> to create unique dedup keys.
        JobPosting(
            title="AI Training - Coding Expert",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.dataannotation.tech/#coding",
            source="DataAnnotation.tech Careers",
            compensation="$20-60+/hr (up to $100+/hr for specialists)",
            description="Coding expert to train AI on programming concepts, code quality, and best practices.",
            requirements=[
                "Advanced coding expertise",
                "Programming language proficiency"
            ],
            responsibilities=[
                "Train on coding",
                "Evaluate code quality",
                "Provide programming feedback"
            ],
            tags=["Coding", "Programming", "AI Training", "Coding Expert", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="AI Training - STEM Expert",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.dataannotation.tech/#stem",
            source="DataAnnotation.tech Careers",
            compensation="$20-60+/hr",
            description="STEM expert to train AI on science, technology, engineering, and mathematics.",
            requirements=[
                "STEM expertise",
                "Advanced degree (MS/PhD) or equivalent"
            ],
            responsibilities=[
                "Train on STEM",
                "Evaluate technical accuracy",
                "Provide STEM feedback"
            ],
            tags=["STEM", "Science", "Technology", "Engineering", "Math", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="AI Training - Writing Expert",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.dataannotation.tech/#writing",
            source="DataAnnotation.tech Careers",
            compensation="$20-60+/hr",
            description="Writing expert to train AI on writing quality, style, clarity, and composition.",
            requirements=[
                "Professional writing expertise",
                "Writing evaluation skills"
            ],
            responsibilities=[
                "Train on writing",
                "Evaluate writing quality",
                "Provide feedback"
            ],
            tags=["Writing", "Content", "AI Training", "Writing Expert", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="AI Training - Law Expert",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.dataannotation.tech/#law",
            source="DataAnnotation.tech Careers",
            compensation="$20-60+/hr",
            description="Law expert to train AI on legal concepts, statutes, and legal reasoning.",
            requirements=[
                "Legal expertise",
                "Law degree or equivalent"
            ],
            responsibilities=[
                "Train on legal concepts",
                "Evaluate legal accuracy",
                "Provide legal feedback"
            ],
            tags=["Legal", "Law", "AI Training", "Law Expert", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="AI Training - Medical Expert",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.dataannotation.tech/#medical",
            source="DataAnnotation.tech Careers",
            compensation="$20-60+/hr",
            description="Medical expert to train AI on medical knowledge, procedures, and healthcare.",
            requirements=[
                "Medical expertise",
                "MD or equivalent medical background"
            ],
            responsibilities=[
                "Train on medical concepts",
                "Evaluate medical accuracy",
                "Provide medical feedback"
            ],
            tags=["Medical", "Healthcare", "AI Training", "Medical Expert", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="FT/PT Remote AI Prompt Engineering & Evaluation",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://weworkremotely.com/remote-jobs/dataannotation-tech-ft-pt-remote-ai-prompt-engineering-evaluation-will-train",
            source="WeWorkRemotely",
            job_type="Full-Time/Part-Time",
            description="AI prompt engineering and evaluation role. Full-time or part-time available. Training provided.",
            requirements=[
                "AI knowledge",
                "Prompt engineering skills"
            ],
            responsibilities=[
                "Create prompts",
                "Evaluate AI responses",
                "Improve prompt quality"
            ],
            tags=["Prompt Engineering", "Evaluation", "AI Training", "Surge AI + DataAnnotation"]
        ),
        # Juji (63)
        JobPosting(
            title="AI Trainer",
            company="Juji",
            location="San Jose/Remote",
            url="https://juji.io/career/",
            source="Juji Careers",
            description="AI trainer focusing on natural language understanding and conversation writing.",
            requirements=[
                "NLU expertise",
                "Conversation writing skills"
            ],
            responsibilities=[
                "Train on NLU",
                "Write conversations",
                "Provide feedback"
            ],
            tags=["NLU", "Conversation Writing", "AI Training", "Juji"]
        ),
        # Handshake (64)
        JobPosting(
            title="AI Training Fellow",
            company="Handshake",
            location="Remote",
            url="https://joinhandshake.com/fellowship-program/",
            source="Handshake",
            compensation="$22-30/hr generalist, $30-150/hr Master's/PhD, $175-300+/hr specialists",
            job_type="Fellowship",
            description="AI training fellowship with tiered compensation based on expertise level. Generalist, Master's/PhD, or specialist tracks available.",
            requirements=[
                "AI training capability",
                "Relevant expertise for chosen track"
            ],
            responsibilities=[
                "Train AI models",
                "Annotate data",
                "Provide expertise"
            ],
            tags=["Fellowship", "AI Training", "Tiered Pay", "Handshake"]
        ),
        # Mercor (65)
        JobPosting(
            title="AI Model Specialist",
            company="Mercor",
            location="Remote",
            url="https://www.mercor.com/careers/",
            source="Mercor Careers",
            compensation="$16/hr entry, $70/hr expert, $70-200+/hr engineering",
            description="AI model specialist with tiered compensation. Entry level $16/hr, experts $70/hr, engineering specialists $70-200+/hr.",
            requirements=[
                "AI/ML knowledge",
                "Model understanding"
            ],
            responsibilities=[
                "Work on AI models",
                "Provide expertise",
                "Train models"
            ],
            tags=["AI Models", "Tiered Pay", "Freelance", "Mercor"]
        ),
        # Collide Capital (66)
        JobPosting(
            title="Data Entry Agent AI Trainer",
            company="Collide Capital",
            location="Remote",
            url="https://jobs.collidecap.com/companies/linkedin-3-496a6401-d340-42f0-8d7a-d62312522866/jobs/68369524-data-entry-agent-ai-trainer-40-50-hour",
            source="Collide Capital Job Board",
            compensation="$40-50/hour",
            description="Data entry agent AI trainer role at Collide Capital.",
            requirements=[
                "Data entry expertise",
                "AI training knowledge"
            ],
            responsibilities=[
                "Train AI on data entry",
                "Annotate examples",
                "Provide feedback"
            ],
            tags=["Data Entry", "AI Training", "Collide Capital"]
        ),
        # CloudDevs (67)
        JobPosting(
            title="AI (LLM) Fullstack Engineer",
            company="CloudDevs",
            location="Remote",
            url="https://weworkremotely.com/remote-jobs/clouddevs-ai-llm-fullstack-engineer-3",
            source="WeWorkRemotely",
            description="Fullstack engineer role working with AI and LLMs for CloudDevs.",
            requirements=[
                "Fullstack engineering expertise",
                "LLM knowledge"
            ],
            responsibilities=[
                "Develop fullstack solutions",
                "Work with LLMs",
                "Engineer AI systems"
            ],
            tags=["Fullstack Engineer", "LLM", "AI Engineering", "CloudDevs"]
        ),
        # BUKI (68-73)
        JobPosting(
            title="AI / Python Programming Tutor",
            company="BUKI",
            location="Las Vegas",
            url="https://www.talent.com/view?id=194a9f348729",
            source="Talent.com",
            compensation="$70-100/hr",
            description="AI and Python programming tutor in Las Vegas area.",
            requirements=[
                "Python expertise",
                "AI knowledge",
                "Teaching ability"
            ],
            responsibilities=[
                "Teach Python",
                "Train on AI",
                "Provide tutoring"
            ],
            tags=["Python", "Programming", "AI", "Tutoring", "BUKI"]
        ),
        JobPosting(
            title="Information Technology Tutor",
            company="BUKI",
            location="San Diego",
            url="https://www.talent.com/view?id=3b0fa03d1d6b",
            source="Talent.com",
            description="Information technology tutor in San Diego area.",
            requirements=[
                "IT expertise",
                "Teaching skills"
            ],
            responsibilities=[
                "Teach IT concepts",
                "Provide tutoring",
                "Support learning"
            ],
            tags=["IT", "Tutoring", "San Diego", "BUKI"]
        ),
        JobPosting(
            title="Information Technology Tutor",
            company="BUKI",
            location="El Paso",
            url="https://www.talent.com/view?id=8540905c3e57",
            source="Talent.com",
            description="Information technology tutor in El Paso area.",
            requirements=[
                "IT expertise",
                "Teaching skills"
            ],
            responsibilities=[
                "Teach IT concepts",
                "Provide tutoring",
                "Support learning"
            ],
            tags=["IT", "Tutoring", "El Paso", "BUKI"]
        ),
        JobPosting(
            title="Information Technology Tutor",
            company="BUKI",
            location="Los Angeles",
            url="https://www.talent.com/view?id=eb60827b49f2",
            source="Talent.com",
            description="Information technology tutor in Los Angeles area.",
            requirements=[
                "IT expertise",
                "Teaching skills"
            ],
            responsibilities=[
                "Teach IT concepts",
                "Provide tutoring",
                "Support learning"
            ],
            tags=["IT", "Tutoring", "Los Angeles", "BUKI"]
        ),
        JobPosting(
            title="Writing Tutor",
            company="BUKI",
            location="San Diego",
            url="https://www.talent.com/view?id=4211a7159c7e",
            source="Talent.com",
            description="Writing tutor in San Diego area.",
            requirements=[
                "Writing expertise",
                "Teaching skills"
            ],
            responsibilities=[
                "Teach writing",
                "Provide tutoring",
                "Support learning"
            ],
            tags=["Writing", "Tutoring", "San Diego", "BUKI"]
        ),
        JobPosting(
            title="Writing Tutor",
            company="BUKI",
            location="Fresno",
            url="https://www.talent.com/view?id=3ecb087d5f5c",
            source="Talent.com",
            description="Writing tutor in Fresno area.",
            requirements=[
                "Writing expertise",
                "Teaching skills"
            ],
            responsibilities=[
                "Teach writing",
                "Provide tutoring",
                "Support learning"
            ],
            tags=["Writing", "Tutoring", "Fresno", "BUKI"]
        ),
        # Outlier (74-75)
        JobPosting(
            title="AI Trainer - Generalist",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/#generalist",
            source="Outlier (Scale AI)",
            compensation="$14-30/hr",
            job_type="Freelance",
            description="Generalist AI trainer role with 700K+ contributors globally. Flexible part-time work.",
            requirements=[
                "AI knowledge",
                "Attention to detail"
            ],
            responsibilities=[
                "Train AI models",
                "Annotate data",
                "Provide feedback"
            ],
            tags=["AI Training", "Generalist", "Freelance", "Outlier", "Scale AI"]
        ),
        JobPosting(
            title="AI Trainer - Coding Specialist",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/#coding-specialist",
            source="Outlier (Scale AI)",
            compensation="$25-50/hr",
            job_type="Freelance",
            description="Coding specialist AI trainer role. Freelance work with higher compensation for technical expertise.",
            requirements=[
                "Advanced coding expertise",
                "Programming proficiency"
            ],
            responsibilities=[
                "Train on coding",
                "Evaluate code",
                "Provide feedback"
            ],
            tags=["Coding", "AI Training", "Specialist", "Freelance", "Outlier"]
        ),
        # Scale AI (76)
        JobPosting(
            title="Data Operations - AI Training",
            company="Scale AI",
            location="San Francisco/Remote",
            url="https://scale.com/careers",
            source="Scale AI Careers",
            job_type="Full-Time",
            description="Data operations role supporting AI training initiatives. Full-time position in San Francisco or remote.",
            requirements=[
                "Data operations experience",
                "AI training knowledge"
            ],
            responsibilities=[
                "Manage data operations",
                "Support AI training",
                "Ensure data quality"
            ],
            tags=["Data Operations", "AI Training", "Full-Time", "Scale AI"]
        ),
        # Remotasks (77)
        JobPosting(
            title="AI Data Annotator - Computer Vision & NLP",
            company="Scale AI + Remotasks",
            location="Remote",
            url="https://www.remotasks.com/",
            source="Remotasks (Scale AI)",
            compensation="$3-20+/hr",
            job_type="Freelance",
            description="AI data annotator role covering computer vision and NLP tasks. Flexible freelance work.",
            requirements=[
                "Computer vision knowledge OR NLP expertise"
            ],
            responsibilities=[
                "Annotate images/text",
                "Label data",
                "Ensure quality"
            ],
            tags=["Data Annotation", "Computer Vision", "NLP", "Freelance", "Remotasks"]
        ),
        # Mindrift (78-79)
        JobPosting(
            title="Freelance English Writer - AI Trainer",
            company="Toloka + Mindrift",
            location="Remote",
            url="https://mindrift.ai/apply#writer",
            source="Mindrift (Toloka)",
            compensation="$15-100+/hr",
            job_type="Freelance",
            description="English writer for AI training. Freelance role with variable compensation based on expertise.",
            requirements=[
                "Professional writing expertise",
                "English fluency"
            ],
            responsibilities=[
                "Write training content",
                "Train AI models",
                "Evaluate responses"
            ],
            tags=["Writing", "English", "Freelance", "AI Training", "Mindrift"]
        ),
        JobPosting(
            title="AI Trainer - Domain Expert",
            company="Toloka + Mindrift",
            location="Remote",
            url="https://mindrift.ai/apply#domain-expert",
            source="Mindrift (Toloka)",
            compensation="$40-100+/hr",
            job_type="Freelance",
            description="Domain expert AI trainer in coding, finance, law, or medicine. Freelance work with higher compensation.",
            requirements=[
                "Expert-level domain knowledge",
                "Coding OR Finance OR Law OR Medicine expertise"
            ],
            responsibilities=[
                "Train on domain expertise",
                "Evaluate responses",
                "Provide feedback"
            ],
            tags=["Domain Expert", "Coding", "Finance", "Law", "Medicine", "Freelance", "Mindrift"]
        ),
        # Toloka (80)
        JobPosting(
            title="AI Trainer - Freelance Data Annotator",
            company="Nebius Group + Toloka",
            location="Remote",
            url="https://toloka.ai/annotator_apply",
            source="Toloka",
            job_type="Freelance",
            description="Freelance data annotator through Toloka global crowdsourcing platform.",
            requirements=[
                "Data annotation capability"
            ],
            responsibilities=[
                "Annotate data",
                "Complete tasks",
                "Ensure quality"
            ],
            tags=["Data Annotation", "Crowdsourcing", "Freelance", "Global", "Toloka"]
        ),
        # LXT (81)
        JobPosting(
            title="AI Data Annotation Specialist",
            company="LXT",
            location="Remote",
            url="https://www.lxt.ai/jobs/",
            source="LXT Careers",
            compensation="~$65.77/hr average",
            description="AI data annotation specialist covering language, speech, and localization work. Average hourly rate ~$65.77.",
            requirements=[
                "Language expertise",
                "Speech knowledge OR Localization experience"
            ],
            responsibilities=[
                "Annotate data",
                "Support language work",
                "Ensure quality"
            ],
            tags=["Data Annotation", "Language", "Speech", "Localization", "LXT"]
        ),
        # Anuttacon (82-83)
        JobPosting(
            title="AI Trainer, LLM",
            company="Anuttacon",
            location="Remote",
            url="https://weworkremotely.com/remote-jobs/anuttacon-ai-trainer-llm",
            source="WeWorkRemotely",
            description="AI trainer specializing in large language models (LLMs).",
            requirements=[
                "LLM expertise",
                "AI training knowledge"
            ],
            responsibilities=[
                "Train LLMs",
                "Annotate data",
                "Provide feedback"
            ],
            tags=["LLM", "AI Training", "Anuttacon"]
        ),
        JobPosting(
            title="Humanized AI Trainer",
            company="Anuttacon",
            location="Remote",
            url="https://nodesk.co/remote-jobs/anuttacon-humanized-ai-trainer/",
            source="NoDesk",
            description="Humanized AI trainer focusing on human-like responses and interactions.",
            requirements=[
                "Human-centered AI knowledge",
                "Interaction expertise"
            ],
            responsibilities=[
                "Train on human interactions",
                "Evaluate responses",
                "Ensure humanization"
            ],
            tags=["Humanized AI", "AI Training", "Anuttacon"]
        ),
        # Welocalize (84)
        JobPosting(
            title="Search Quality Rater / AI Trainer",
            company="Welocalize",
            location="Remote",
            url="https://www.welocalize.com/",
            source="Welocalize Careers",
            description="Search quality rater and AI trainer role focusing on multilingual search evaluation.",
            requirements=[
                "Multilingual capability",
                "Search evaluation skills"
            ],
            responsibilities=[
                "Rate search quality",
                "Evaluate multilingual content",
                "Train AI systems"
            ],
            tags=["Search Quality", "Multilingual", "Evaluation", "Welocalize"]
        ),
        # RWS TrainAI (85)
        JobPosting(
            title="AI Data Specialist - TrainAI Community",
            company="RWS TrainAI",
            location="Remote",
            url="https://www.rws.com/artificial-intelligence/train-ai-data-services/trainai-community/",
            source="RWS TrainAI",
            compensation="$4-20/hr",
            job_type="Freelance",
            description="AI data specialist in TrainAI Community. Online rater, annotator, or search evaluator roles available.",
            requirements=[
                "Data annotation OR rating OR evaluation capability"
            ],
            responsibilities=[
                "Rate or annotate data",
                "Evaluate search results",
                "Complete assessments"
            ],
            tags=["Data Annotation", "Search Evaluation", "Online Rater", "RWS", "TrainAI"]
        ),
        # Appen (86)
        JobPosting(
            title="AI Training Data Contributor",
            company="Appen",
            location="Remote",
            url="https://appen.com/",
            source="Appen",
            job_type="Freelance",
            description="Contribute to AI training data through crowdwork and specialized projects on the Appen platform.",
            requirements=[
                "Data annotation capability"
            ],
            responsibilities=[
                "Contribute training data",
                "Complete crowdwork",
                "Support projects"
            ],
            tags=["Data Annotation", "Crowdwork", "Freelance", "Appen"]
        ),
        # Alignerr (87)
        JobPosting(
            title="AI Alignment Specialist",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://alignerr.com/",
            source="Alignerr (Labelbox)",
            description="AI alignment specialist using cognitive labeling to ensure ethical AI training and development.",
            requirements=[
                "AI alignment knowledge",
                "Cognitive labeling expertise"
            ],
            responsibilities=[
                "Label for alignment",
                "Ensure ethical training",
                "Provide feedback"
            ],
            tags=["AI Alignment", "Cognitive Labeling", "Ethics", "Labelbox", "Alignerr"]
        ),
        # Gloz (88)
        JobPosting(
            title="AI Training - Language Specialist",
            company="Gloz",
            location="Remote",
            url="https://gloz.ai/",
            source="Gloz",
            description="Language specialist for LLM evaluation and training. Focus on language-based AI assessment.",
            requirements=[
                "Language expertise",
                "LLM knowledge"
            ],
            responsibilities=[
                "Evaluate language quality",
                "Train language models",
                "Provide feedback"
            ],
            tags=["Language", "LLM Evaluation", "AI Training", "Gloz"]
        ),
        # OpenTrain AI (89)
        JobPosting(
            title="Freelance AI Trainer / Data Labeler",
            company="OpenTrain AI",
            location="Remote",
            url="https://www.opentrain.ai/become-freelancer/",
            source="OpenTrain AI",
            job_type="Freelance",
            description="Freelance AI trainer and data labeler. Work on RLHF, red teaming, and data labeling projects.",
            requirements=[
                "AI training OR data labeling experience",
                "RLHF OR red teaming knowledge helpful"
            ],
            responsibilities=[
                "Train AI models",
                "Label data",
                "Red team systems",
                "RLHF work"
            ],
            tags=["AI Training", "Data Labeling", "RLHF", "Red Teaming", "Freelance", "OpenTrain"]
        ),
        # Embedding VC (90)
        JobPosting(
            title="AI Video Generation Specialist",
            company="Embedding VC",
            location="Remote",
            url="https://wellfound.com/company/embedding-vc/jobs",
            source="Wellfound",
            description="Video generation specialist for AI training. 36+ positions available for various expertise levels.",
            requirements=[
                "Video generation expertise",
                "Video knowledge"
            ],
            responsibilities=[
                "Train on video generation",
                "Evaluate video quality",
                "Provide feedback"
            ],
            tags=["Video Generation", "AI Training", "Specialist", "Embedding VC"]
        ),
        # Recruiting from Scratch (91)
        JobPosting(
            title="AI/ML Data Annotation Specialist",
            company="Recruiting from Scratch",
            location="Remote",
            url="https://www.recruitingfromscratch.com/",
            source="Recruiting from Scratch",
            description="Data annotation specialist for AI/ML projects. Staffing and placement services.",
            requirements=[
                "Data annotation experience",
                "ML knowledge"
            ],
            responsibilities=[
                "Annotate data",
                "Train models",
                "Quality assurance"
            ],
            tags=["Data Annotation", "AI/ML", "Staffing", "Recruiting from Scratch"]
        ),
        # Braintrust (92)
        JobPosting(
            title="Human Data Annotator for AI Training",
            company="Braintrust",
            location="Remote",
            url="https://www.usebraintrust.com/human-data",
            source="Braintrust",
            job_type="Freelance",
            description="Human data annotator for AI model training through the Braintrust platform.",
            requirements=[
                "Data annotation capability"
            ],
            responsibilities=[
                "Annotate training data",
                "Evaluate quality",
                "Provide feedback"
            ],
            tags=["Data Annotation", "Human Data", "Freelance", "Braintrust"]
        ),
        # TELUS Digital (93-94)
        JobPosting(
            title="AI Data Annotation Specialist - English",
            company="TELUS Digital",
            location="Remote",
            url="https://www.telusinternational.ai/opportunities",
            source="TELUS Digital",
            job_type="Freelance",
            compensation="Flexible hourly",
            description="AI data annotation specialist for English language. Global freelance work with flexible hours.",
            requirements=[
                "English proficiency",
                "Data annotation capability"
            ],
            responsibilities=[
                "Annotate data",
                "Label content",
                "Ensure quality"
            ],
            tags=["Data Annotation", "English", "Freelance", "Flexible Hours", "TELUS Digital"]
        ),
        JobPosting(
            title="Personalized Internet Assessor / Search Evaluator",
            company="TELUS Digital",
            location="Remote",
            url="https://www.telusdigital.com/careers",
            source="TELUS Digital",
            compensation="$12-40/hr",
            description="Internet assessor and search evaluator. Rate search results and provide feedback on search quality.",
            requirements=[
                "Internet knowledge",
                "Search evaluation skills"
            ],
            responsibilities=[
                "Evaluate search results",
                "Rate internet content",
                "Provide quality feedback"
            ],
            tags=["Search Evaluation", "Internet Assessment", "TELUS Digital"]
        ),
        # Innodata (95-98)
        JobPosting(
            title="Generative AI Associate (English)",
            company="Innodata",
            location="Remote",
            url="https://careers.innodata.com/",
            source="Innodata Careers",
            description="Generative AI associate role covering evaluation, annotation, and classification tasks.",
            requirements=[
                "English proficiency",
                "AI knowledge"
            ],
            responsibilities=[
                "Evaluate AI outputs",
                "Annotate content",
                "Classify data"
            ],
            tags=["Generative AI", "Evaluation", "Annotation", "Classification", "Innodata"]
        ),
        JobPosting(
            title="Generative AI Associate - Red Teaming Specialist",
            company="Innodata",
            location="Remote",
            url="https://innodatainc.recruitee.com/#redteam",
            source="Innodata Careers",
            description="Red teaming specialist for adversarial testing of AI systems. Test for vulnerabilities and biases.",
            requirements=[
                "Red teaming experience",
                "Adversarial testing knowledge"
            ],
            responsibilities=[
                "Red team AI systems",
                "Test for vulnerabilities",
                "Identify biases"
            ],
            tags=["Red Teaming", "Adversarial Testing", "Security", "Innodata"]
        ),
        JobPosting(
            title="AI Rewriting Specialist",
            company="Innodata",
            location="Remote",
            url="https://innodatainc.recruitee.com/#rewriting",
            source="Innodata Careers",
            description="Rewriting specialist to improve and refine AI-generated content.",
            requirements=[
                "Writing expertise",
                "Content refinement skills"
            ],
            responsibilities=[
                "Rewrite content",
                "Improve quality",
                "Provide feedback"
            ],
            tags=["Rewriting", "Writing", "Content Improvement", "Innodata"]
        ),
        JobPosting(
            title="Data Annotator",
            company="Innodata",
            location="Remote",
            url="https://innodatainc.recruitee.com/#annotator",
            source="Innodata Careers",
            compensation="₹350/hr (India)",
            description="Data annotator to review and label digital data. Work available in India with competitive hourly rate.",
            requirements=[
                "Data annotation capability"
            ],
            responsibilities=[
                "Review digital data",
                "Label content",
                "Ensure quality"
            ],
            tags=["Data Annotation", "Labeling", "India", "Innodata"]
        ),
        # iMerit (99-100)
        JobPosting(
            title="AI Trainer - English",
            company="iMerit",
            location="Remote",
            url="https://imerit.net/careers/",
            source="iMerit Careers",
            description="AI trainer for multimodal AI covering text, images, video, and audio. Requires 1+ year annotation experience.",
            requirements=[
                "English proficiency",
                "1+ year annotation experience",
                "Multimodal AI knowledge"
            ],
            responsibilities=[
                "Train multimodal AI",
                "Annotate diverse content",
                "Provide feedback"
            ],
            tags=["AI Training", "Multimodal", "Text", "Images", "Video", "Audio", "iMerit"]
        ),
        JobPosting(
            title="AI Data Annotator",
            company="iMerit",
            location="Remote",
            url="https://jobs.weekday.works/imerit-ai-data-annotator",
            source="Weekday",
            description="AI data annotator role at iMerit.",
            requirements=[
                "Data annotation capability"
            ],
            responsibilities=[
                "Annotate data",
                "Label content",
                "Ensure quality"
            ],
            tags=["Data Annotation", "iMerit"]
        ),
        # Invisible Technologies (101-106)
        JobPosting(
            title="Advanced AI Data Trainer",
            company="Invisible Technologies",
            location="Remote",
            url="https://invisibletech.ai/join-us",
            source="Invisible Technologies",
            compensation="$6-65/hr",
            description="Advanced AI data trainer with 116+ positions available. Compensation ranges from $6-65/hr based on expertise.",
            requirements=[
                "Data training capability",
                "AI knowledge"
            ],
            responsibilities=[
                "Train AI models",
                "Annotate data",
                "Provide feedback"
            ],
            tags=["AI Training", "Data Annotation", "Invisible Technologies"]
        ),
        JobPosting(
            title="Social Media Annotation - Freelance AI Trainer",
            company="Invisible Technologies",
            location="Remote",
            url="https://www.glassdoor.com/company/Invisible-Technologies/",
            source="Glassdoor",
            description="Freelance AI trainer specializing in social media annotation and content evaluation.",
            requirements=[
                "Social media expertise",
                "Content knowledge"
            ],
            responsibilities=[
                "Annotate social media",
                "Evaluate content",
                "Train on social trends"
            ],
            tags=["Social Media", "Annotation", "Freelance", "Invisible Technologies"]
        ),
        JobPosting(
            title="Literature Specialist - Freelance AI Trainer",
            company="Invisible Technologies",
            location="Remote",
            description="Literature specialist for AI training. Freelance role working on literary content.",
            requirements=[
                "Literature expertise",
                "Writing knowledge"
            ],
            responsibilities=[
                "Train on literature",
                "Annotate text",
                "Provide feedback"
            ],
            tags=["Literature", "Writing", "Freelance", "Invisible Technologies"]
        ),
        JobPosting(
            title="Politics & Government Specialist - Freelance AI Trainer",
            company="Invisible Technologies",
            location="Remote",
            description="Politics and government specialist for AI training. Freelance work on political/governmental content.",
            requirements=[
                "Politics/Government expertise",
                "Policy knowledge"
            ],
            responsibilities=[
                "Train on politics",
                "Annotate policy content",
                "Provide feedback"
            ],
            tags=["Politics", "Government", "Policy", "Freelance", "Invisible Technologies"]
        ),
        JobPosting(
            title="Video Game Specialist - Freelance AI Trainer",
            company="Invisible Technologies",
            location="Remote",
            description="Video game specialist for AI training. Freelance role training AI on gaming content.",
            requirements=[
                "Video game expertise",
                "Gaming knowledge"
            ],
            responsibilities=[
                "Train on games",
                "Annotate game content",
                "Provide game feedback"
            ],
            tags=["Video Games", "Gaming", "Freelance", "Invisible Technologies"]
        ),
        JobPosting(
            title="Sports Specialist - Freelance AI Trainer",
            company="Invisible Technologies",
            location="Remote",
            description="Sports specialist for AI training. Freelance work on sports-related content.",
            requirements=[
                "Sports expertise",
                "Sports knowledge"
            ],
            responsibilities=[
                "Train on sports",
                "Annotate sports content",
                "Provide sports feedback"
            ],
            tags=["Sports", "Freelance", "Invisible Technologies"]
        ),
        # Comrise (107)
        JobPosting(
            title="Data Annotation Specialist (Robotics/Computer Vision)",
            company="Comrise",
            location="Remote",
            url="https://www.linkedin.com/jobs/view/data-annotation-specialist-at-comrise-4383244530",
            source="LinkedIn",
            compensation="$20-40/hr, 20-30hrs/week",
            description="Data annotation specialist for robotics and computer vision. Focus on autonomous vehicles and drone data.",
            requirements=[
                "Computer vision expertise",
                "Robotics knowledge",
                "Annotation experience"
            ],
            responsibilities=[
                "Annotate robotics data",
                "Label computer vision datasets",
                "Support autonomous systems"
            ],
            tags=["Computer Vision", "Robotics", "Autonomous Vehicles", "Drones", "Comrise"]
        ),
        # Surge AI (108-109)
        JobPosting(
            title="AI Data Trainer / Annotator",
            company="Surge AI",
            location="Remote",
            url="https://surgehq.ai/workforce",
            source="Surge AI",
            description="Task-based AI data trainer and annotator role. Flexible freelance work on specific projects.",
            requirements=[
                "Data annotation capability",
                "Task-based work capability"
            ],
            responsibilities=[
                "Train AI models",
                "Annotate data",
                "Complete tasks"
            ],
            tags=["AI Training", "Data Annotation", "Task-Based", "Freelance", "Surge AI"]
        ),
        JobPosting(
            title="Engineering/Research/Ops roles",
            company="Surge AI",
            location="Remote",
            url="https://surgehq.ai/careers",
            source="Surge AI Careers",
            description="Engineering, research, and operations roles at Surge AI. Full-time career opportunities.",
            requirements=[
                "Engineering OR Research OR Operations expertise"
            ],
            responsibilities=[
                "Engineer systems",
                "Conduct research",
                "Manage operations"
            ],
            tags=["Engineering", "Research", "Operations", "Full-Time", "Surge AI"]
        ),
        # Sama (110)
        JobPosting(
            title="Data Annotation Specialist",
            company="Sama",
            location="Remote",
            url="https://www.sama.com/careers",
            source="Sama Careers",
            description="Data annotation specialist for image and video tagging, segmentation, and labeling.",
            requirements=[
                "Attention to detail",
                "Image/video knowledge"
            ],
            responsibilities=[
                "Tag images/video",
                "Segment content",
                "Label accurately"
            ],
            tags=["Data Annotation", "Image Tagging", "Video Tagging", "Segmentation", "Sama"]
        ),
        # CloudFactory (111)
        JobPosting(
            title="Data Annotation Specialist",
            company="CloudFactory",
            location="Remote",
            url="https://www.cloudfactory.com/careers",
            source="CloudFactory Careers",
            description="Data annotation specialist for robotics and drone vision. Dedicated team-based work.",
            requirements=[
                "Robotics/Vision expertise",
                "Annotation experience"
            ],
            responsibilities=[
                "Annotate robotics data",
                "Label vision datasets",
                "Support team projects"
            ],
            tags=["Data Annotation", "Robotics", "Drone Vision", "Computer Vision", "CloudFactory"]
        ),
        # Meridial/Invisible Technologies (112)
        JobPosting(
            title="Domain Expert - AI Training",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://www.meridial.ai",
            source="Meridial (Invisible Technologies)",
            description="Domain expert for AI training in law, STEM, finance, coding, linguistics, or safety.",
            requirements=[
                "Expert-level domain knowledge",
                "Law OR STEM OR Finance OR Coding OR Linguistics OR Safety expertise"
            ],
            responsibilities=[
                "Train on domain expertise",
                "Evaluate responses",
                "Provide specialized feedback"
            ],
            tags=["Domain Expert", "Law", "STEM", "Finance", "Coding", "Linguistics", "Safety", "Meridial"]
        ),

        # =====================================================
        # Meridial / Invisible Technologies (owned by Perplexity) — Specialist AI Trainer Roles (113-140)
        # =====================================================
        JobPosting(
            title="Statistics Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=d907a3a212ba0a30",
            source="Indeed",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Seeking statistics specialists with expertise in probability theory, statistical modeling, data analysis, hypothesis testing, regression analysis, multivariate statistics, Bayesian inference, time series analysis, experimental design, and machine learning algorithms.",
            requirements=["Master's or PhD in statistics, data science, mathematics preferred"],
            responsibilities=["Train AI models on statistics concepts", "Evaluate AI responses for statistical accuracy", "Provide expert feedback"],
            tags=["Statistics", "AI Trainer", "Freelance", "Meridial", "Data Science"]
        ),
        JobPosting(
            title="Accounting Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=14ec861984bb9f5e",
            source="Indeed",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Seeking accounting specialists. CPA licensure, experience with financial audits, tax preparation, or ERP systems like SAP or Oracle a plus.",
            requirements=["Bachelor's or Master's in accounting, finance, or related field", "CPA preferred"],
            responsibilities=["Train AI models on accounting concepts", "Evaluate financial reasoning", "Provide expert feedback"],
            tags=["Accounting", "Finance", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Sports Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=45f0ab4bb7aa012b",
            source="Indeed",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Sports domain expert for AI training. Deep knowledge of sports analytics, rules, history, and statistics.",
            requirements=["Deep sports domain knowledge"],
            responsibilities=["Train AI models on sports knowledge", "Evaluate sports-related AI responses", "Provide expert feedback"],
            tags=["Sports", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Wetlab Protocol Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=2ebf6a380c1418e8",
            source="Indeed",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Seeking wetlab protocol specialists with expertise in laboratory procedures, experimental biology, chemistry protocols, and scientific methodology.",
            requirements=["Advanced degree in biology, chemistry, or related lab science"],
            responsibilities=["Train AI on lab protocols", "Evaluate scientific accuracy", "Provide expert feedback"],
            tags=["Wetlab", "Biology", "Chemistry", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Computer Science Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/agency/jobs/4577596101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Seeking CS specialists with expertise in algorithms, data structures, machine learning, and related fields.",
            requirements=["Strong CS background", "Expertise in algorithms, data structures, ML"],
            responsibilities=["Train AI models on CS concepts", "Evaluate code and technical reasoning", "Provide expert feedback"],
            tags=["Computer Science", "AI Trainer", "Freelance", "Meridial", "Algorithms"]
        ),
        JobPosting(
            title="C# Coding Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/agency/jobs/4633533101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Seeking C# coding specialists to train and evaluate AI models on C# programming, .NET framework, and software development.",
            requirements=["Strong C# and .NET expertise"],
            responsibilities=["Train AI on C# programming", "Evaluate code quality", "Provide expert feedback"],
            tags=["C#", ".NET", "Coding", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Data Science Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.greenhouse.io/agency/jobs/4589022101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Seeking data science specialists with expertise in machine learning, statistical modeling, data engineering, and related fields.",
            requirements=["Strong data science background", "ML and statistical modeling expertise"],
            responsibilities=["Train AI models on data science", "Evaluate analytical reasoning", "Provide expert feedback"],
            tags=["Data Science", "ML", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="STEM Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/agency/jobs/4607005101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Seeking STEM specialists for AI training projects covering science, technology, engineering, and mathematics domains.",
            requirements=["Advanced STEM degree preferred"],
            responsibilities=["Train AI on STEM topics", "Evaluate technical accuracy", "Provide expert feedback"],
            tags=["STEM", "Science", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Science Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/agency/jobs/4607789101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Seeking science specialists for AI training across biology, physics, chemistry, earth sciences, and related fields.",
            requirements=["Advanced science degree"],
            responsibilities=["Train AI on science concepts", "Evaluate scientific reasoning", "Provide expert feedback"],
            tags=["Science", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Computer & Information Systems Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/agency/jobs/4605300101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Seeking Computer & Information Systems specialists for AI training in networking, databases, systems architecture, and IT.",
            requirements=["CIS or IT background"],
            responsibilities=["Train AI on IT systems", "Evaluate technical responses", "Provide expert feedback"],
            tags=["IT", "Information Systems", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="General Operations Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/agency/jobs/4605323101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="General operations specialist for AI training. Covers business operations, logistics, project management, and process optimization.",
            requirements=["Operations or business background"],
            responsibilities=["Train AI on operations topics", "Evaluate business reasoning", "Provide expert feedback"],
            tags=["Operations", "Business", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Customer Service Representative Specialist - AI Trainer",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/agency/jobs/4605302101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Train AI models on customer service best practices, communication, conflict resolution, and support workflows.",
            requirements=["Customer service experience"],
            responsibilities=["Train AI on customer service", "Evaluate conversational quality", "Provide expert feedback"],
            tags=["Customer Service", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Training and Development Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://boards.greenhouse.io/embed/job_app?token=4605461101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Training and development specialist for AI training projects. Covers instructional design, learning theory, and corporate training.",
            requirements=["Training/L&D background"],
            responsibilities=["Train AI on T&D concepts", "Evaluate instructional quality", "Provide expert feedback"],
            tags=["Training", "L&D", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Architecture & Construction Documentation Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.greenhouse.io/agency/jobs/4737146101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Seeking architecture and construction documentation specialists. Expertise in building codes, blueprints, construction management, and architectural design.",
            requirements=["Architecture or construction background"],
            responsibilities=["Train AI on architecture/construction", "Evaluate technical documentation", "Provide expert feedback"],
            tags=["Architecture", "Construction", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Product Deep Research Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.greenhouse.io/agency/jobs/4822831101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Training and evaluating an AI model's capacity to perform structured, high-nuance analysis on specific products.",
            requirements=["Research and analytical skills", "Product analysis experience"],
            responsibilities=["Train AI on product research", "Evaluate deep research outputs", "Provide expert feedback"],
            tags=["Product Research", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="Literature Specialist - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/agency/jobs/4784881101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Literature specialist for AI training. Expertise in literary analysis, genre conventions, narrative structure, and critical theory.",
            requirements=["Advanced literature degree preferred"],
            responsibilities=["Train AI on literary concepts", "Evaluate creative and analytical outputs", "Provide expert feedback"],
            tags=["Literature", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="AI Generalist (No Experience Required) - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote (US Only)",
            url="https://job-boards.eu.greenhouse.io/agency/jobs/4677212101",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="Entry-level AI training opportunity. No prior AI experience needed. Suitable for people early in their academic journey or exploring specialties.",
            requirements=["US-based", "No experience required"],
            responsibilities=["Interact with AI systems", "Review model outputs", "Follow task guidelines", "Contribute to training data"],
            tags=["Entry Level", "Generalist", "AI Trainer", "Freelance", "Meridial"]
        ),
        JobPosting(
            title="English Language Specialist (US Only) - Freelance AI Trainer Project",
            company="Invisible Technologies + Meridial",
            location="Remote (US Only)",
            url="https://job-boards.eu.greenhouse.io/agency/jobs/4677212101#english",
            source="Invisible Agency (Greenhouse)",
            compensation="$8-$65/hr",
            job_type="Freelance/Contract",
            description="English language specialist for AI training. Focus on grammar, composition, rhetoric, and language pedagogy.",
            requirements=["English language expertise", "US-based"],
            responsibilities=["Train AI on English language", "Evaluate linguistic quality", "Provide expert feedback"],
            tags=["English", "Language", "AI Trainer", "Freelance", "Meridial"]
        ),

        # =====================================================
        # Outlier AI Roles (141-147)
        # =====================================================
        JobPosting(
            title="AI Training Specialist",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/#specialist",
            source="Outlier AI",
            compensation="$14-$30/hr (generalist), up to $50+/hr (specialist)",
            job_type="Freelance/Contract",
            description="Train the next generation of AI as a freelancer. Evaluate AI responses, write prompts, judge outputs, and improve model performance.",
            requirements=["No AI experience required", "Subject matter expertise a plus"],
            responsibilities=["Write and evaluate prompts", "Judge AI responses", "Improve model performance", "Test edge cases"],
            tags=["AI Trainer", "Freelance", "Outlier AI", "RLHF"]
        ),
        JobPosting(
            title="AI Writing Evaluator",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/#writing",
            source="Outlier AI",
            compensation="$14-$30/hr",
            job_type="Freelance/Contract",
            description="Evaluate AI-generated writing for quality, accuracy, tone, and helpfulness. Provide feedback to improve model outputs.",
            requirements=["Strong writing skills", "Attention to detail"],
            responsibilities=["Evaluate AI writing quality", "Rate response helpfulness", "Provide improvement feedback"],
            tags=["Writing", "AI Evaluator", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="AI Math Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/#math",
            source="Outlier AI",
            compensation="$20-$50/hr",
            job_type="Freelance/Contract",
            description="Train AI models on mathematical reasoning, problem-solving, and step-by-step explanations.",
            requirements=["Strong math background", "Degree in mathematics, engineering, or related field preferred"],
            responsibilities=["Train AI on math reasoning", "Evaluate mathematical accuracy", "Write math problems and solutions"],
            tags=["Math", "STEM", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="AI Coding Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/#coding",
            source="Outlier AI",
            compensation="$25-$50/hr",
            job_type="Freelance/Contract",
            description="Train AI models on programming, code generation, debugging, and software engineering best practices.",
            requirements=["4+ years software engineering experience preferred", "Proficiency in Python, JavaScript, or other languages"],
            responsibilities=["Evaluate AI-generated code", "Write coding challenges", "Test code correctness"],
            tags=["Coding", "Software Engineering", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="AI Prompt Writer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/#promptwriter",
            source="Outlier AI",
            compensation="$14-$30/hr",
            job_type="Freelance/Contract",
            description="Write creative, challenging prompts to test and improve AI model capabilities across diverse topics.",
            requirements=["Creativity", "Strong writing skills"],
            responsibilities=["Write diverse prompts", "Test model capabilities", "Identify edge cases"],
            tags=["Prompt Writing", "AI Trainer", "Freelance", "Outlier AI"]
        ),

        # --- Outlier Domain Expert Roles (discovered from outlier.ai/experts) ---
        JobPosting(
            title="Physics Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/physics",
            source="Outlier AI",
            compensation="Up to $80/hr",
            job_type="Freelance/Contract",
            description="Create and answer physics questions to train AI models. Review and rank AI model chains of thought for correctness.",
            requirements=["Master's or PhD in Physics or related field"],
            responsibilities=["Craft physics questions", "Evaluate AI reasoning", "Provide expert feedback"],
            tags=["Physics", "STEM", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Chemistry Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/chemistry",
            source="Outlier AI",
            compensation="Up to $60/hr",
            job_type="Freelance/Contract",
            description="Create and answer chemistry-related questions to train AI models. Review and rank AI-generated reasoning.",
            requirements=["Master's or PhD in Chemistry or related field"],
            responsibilities=["Craft chemistry questions", "Evaluate AI reasoning", "Provide expert feedback"],
            tags=["Chemistry", "STEM", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Medicine Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/medicine",
            source="Outlier AI",
            compensation="Up to $80/hr",
            job_type="Freelance/Contract",
            description="Train AI models in medical knowledge. Evaluate AI-generated medical reasoning and provide expert feedback.",
            requirements=["MD, DO, or advanced degree in medical field"],
            responsibilities=["Create medical questions", "Evaluate AI medical reasoning", "Ensure accuracy"],
            tags=["Medicine", "Healthcare", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Law Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/law",
            source="Outlier AI",
            compensation="Up to $75/hr",
            job_type="Freelance/Contract",
            description="Train AI models in legal reasoning. Evaluate AI-generated legal analysis for accuracy and completeness.",
            requirements=["J.D. or equivalent legal degree", "Practicing attorney experience preferred"],
            responsibilities=["Create legal questions", "Evaluate AI legal reasoning", "Provide expert feedback"],
            tags=["Law", "Legal", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Finance Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/finance",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models in financial concepts. Evaluate AI-generated financial analysis and reasoning.",
            requirements=["Bachelor's or Master's in Finance, Economics, or related field"],
            responsibilities=["Create finance questions", "Evaluate AI financial reasoning", "Provide expert feedback"],
            tags=["Finance", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Accounting Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/accounting",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models in accounting knowledge. Evaluate AI-generated accounting responses for accuracy.",
            requirements=["Bachelor's or Master's in Accounting", "CPA preferred"],
            responsibilities=["Create accounting questions", "Evaluate AI responses", "Provide expert feedback"],
            tags=["Accounting", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Biomedical Engineering Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/biomedical-eng",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models in biomedical engineering concepts. Evaluate AI reasoning in biomedical contexts.",
            requirements=["Master's or PhD in Biomedical Engineering or related field"],
            responsibilities=["Create biomedical questions", "Evaluate AI responses", "Provide expert feedback"],
            tags=["Biomedical Engineering", "STEM", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Earth & Environmental Sciences Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/earth-and-environmental-sciences",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models in earth and environmental sciences. Evaluate AI reasoning in geology, ecology, and climate topics.",
            requirements=["Master's or PhD in Earth Science, Environmental Science, or related field"],
            responsibilities=["Create domain questions", "Evaluate AI reasoning", "Provide expert feedback"],
            tags=["Environmental Science", "Earth Science", "STEM", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Civil Engineering Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/civil-engineering",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models in civil engineering concepts. Evaluate AI responses on structural, transportation, and geotechnical topics.",
            requirements=["Bachelor's or Master's in Civil Engineering"],
            responsibilities=["Create engineering questions", "Evaluate AI reasoning", "Provide expert feedback"],
            tags=["Civil Engineering", "STEM", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Electrical Engineering Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/electrical-engineering",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models in electrical engineering. Evaluate AI responses on circuits, signals, power systems, and electronics.",
            requirements=["Bachelor's or Master's in Electrical Engineering"],
            responsibilities=["Create EE questions", "Evaluate AI reasoning", "Provide expert feedback"],
            tags=["Electrical Engineering", "STEM", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Mechanical Engineering Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/mechanical-engineering",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models in mechanical engineering. Evaluate AI responses on thermodynamics, mechanics, and materials.",
            requirements=["Bachelor's or Master's in Mechanical Engineering"],
            responsibilities=["Create ME questions", "Evaluate AI reasoning", "Provide expert feedback"],
            tags=["Mechanical Engineering", "STEM", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Chemical Engineering Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/chemical-engineering",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models in chemical engineering. Evaluate AI responses on process design, reaction engineering, and separations.",
            requirements=["Bachelor's or Master's in Chemical Engineering"],
            responsibilities=["Create ChemE questions", "Evaluate AI reasoning", "Provide expert feedback"],
            tags=["Chemical Engineering", "STEM", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Data Analyst Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/data-analyst",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models in data analysis. Tasks include finding datasets, data cleaning, building statistical models, and communicating findings.",
            requirements=["Experience in data analysis, statistics, or data science"],
            responsibilities=["Create data analysis tasks", "Evaluate AI data work", "Provide expert feedback"],
            tags=["Data Analysis", "Statistics", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="B2B Digital Domain Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/b2b-digital-domain",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models on B2B digital products: SaaS, cloud infrastructure, cybersecurity, marketing automation, CRM platforms.",
            requirements=["Experience with B2B SaaS, enterprise software, or cloud platforms"],
            responsibilities=["Create B2B domain questions", "Evaluate AI responses", "Provide expert feedback"],
            tags=["B2B", "SaaS", "Digital", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="B2C Digital Domain Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/b2c-digital-domain",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models on B2C digital products: streaming services, social media, online gaming, digital news, music platforms.",
            requirements=["Experience with consumer digital products and platforms"],
            responsibilities=["Create B2C domain questions", "Evaluate AI responses", "Provide expert feedback"],
            tags=["B2C", "Consumer", "Digital", "AI Trainer", "Freelance", "Outlier AI"]
        ),
        JobPosting(
            title="Graphic Design Expert - AI Trainer",
            company="Scale AI + Outlier",
            location="Remote",
            url="https://outlier.ai/experts/graphic-design",
            source="Outlier AI",
            compensation="$30-$60/hr",
            job_type="Freelance/Contract",
            description="Train AI models in graphic design. Evaluate AI-generated design critique, layout principles, and visual communication.",
            requirements=["Professional graphic design experience", "Portfolio of work"],
            responsibilities=["Create design-related questions", "Evaluate AI design responses", "Provide expert feedback"],
            tags=["Graphic Design", "Creative", "AI Trainer", "Freelance", "Outlier AI"]
        ),

        # =====================================================
        # Alignerr (Labelbox) Roles (148-150)
        # =====================================================
        JobPosting(
            title="AI Training Expert",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/#expert",
            source="Alignerr (Labelbox)",
            compensation="$15-$150/hr",
            job_type="Freelance/Contract",
            description="Help train AI models through Alignerr, powered by Labelbox. Evaluate and refine AI outputs across various domains.",
            requirements=["Domain expertise in coding, math, writing, or other fields"],
            responsibilities=["Evaluate AI outputs", "Provide human feedback", "Train AI models"],
            tags=["AI Trainer", "Freelance", "Alignerr", "Labelbox", "RLHF"]
        ),
        JobPosting(
            title="AI Coding Expert",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/#coding",
            source="Alignerr (Labelbox)",
            compensation="$30-$150/hr",
            job_type="Freelance/Contract",
            description="Evaluate and improve AI code generation. Requires strong software engineering background.",
            requirements=["4+ years software engineering", "Proficiency in major programming languages"],
            responsibilities=["Evaluate AI-generated code", "Write reference solutions", "Provide coding feedback"],
            tags=["Coding", "AI Expert", "Freelance", "Alignerr", "Labelbox"]
        ),
        JobPosting(
            title="AI Writing Expert",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/#writing",
            source="Alignerr (Labelbox)",
            compensation="$15-$75/hr",
            job_type="Freelance/Contract",
            description="Evaluate and improve AI writing quality. Provide expert feedback on clarity, accuracy, and style.",
            requirements=["Strong writing portfolio", "Editorial experience preferred"],
            responsibilities=["Evaluate AI writing", "Provide editorial feedback", "Train models on writing quality"],
            tags=["Writing", "AI Expert", "Freelance", "Alignerr", "Labelbox"]
        ),

        # =====================================================
        # Centific Roles (151-152)
        # =====================================================
        JobPosting(
            title="AI Trainer (English)",
            company="Centific",
            location="Remote",
            url="https://www.indeed.com/cmp/Centific/jobs",
            source="Indeed",
            compensation="$8-$65/hr",
            job_type="Contract",
            description="Annotate and curate English-language data for a leading AI lab. Collaborating with Centific, a Redmond, WA-based AI data foundry ($60M Series A).",
            requirements=["Basic English fluency", "Attention to detail", "AI data-handling aptitude"],
            responsibilities=["Annotate English-language data", "Curate training datasets", "Follow quality guidelines"],
            tags=["AI Trainer", "English", "Data Annotation", "Centific"]
        ),
        JobPosting(
            title="Data Annotation Specialist",
            company="Centific",
            location="Remote",
            url="https://www.indeed.com/cmp/Centific/jobs#annotation",
            source="Indeed",
            compensation="$10-$40/hr",
            job_type="Contract",
            description="General data annotation role at Centific supporting frontier AI model development.",
            requirements=["Attention to detail", "Ability to follow guidelines"],
            responsibilities=["Label and annotate data", "Quality assurance", "Follow annotation protocols"],
            tags=["Data Annotation", "AI Training", "Centific"]
        ),

        # =====================================================
        # Scale AI / Remotasks Roles (153-155)
        # =====================================================
        JobPosting(
            title="AI Data Trainer",
            company="Scale AI + Remotasks",
            location="Remote",
            url="https://app.remotasks.com/",
            source="Remotasks",
            compensation="$8-$20/hr (basic), $20-$50/hr (specialist)",
            job_type="Freelance/Contract",
            description="AI training and data annotation for Scale AI's Remotasks platform. Tasks include image, video, and LiDAR annotation for computer vision, plus LLM training tasks.",
            requirements=["Computer literacy", "Attention to detail"],
            responsibilities=["Annotate images, video, and LiDAR data", "Evaluate LLM responses", "Complete structured training programs"],
            tags=["AI Trainer", "Data Annotation", "Scale AI", "Remotasks", "Computer Vision"]
        ),
        JobPosting(
            title="LLM Response Evaluator",
            company="Scale AI + Remotasks",
            location="Remote",
            url="https://app.remotasks.com/#llm",
            source="Remotasks",
            compensation="$15-$50/hr",
            job_type="Freelance/Contract",
            description="Evaluate and rank LLM responses for quality, helpfulness, and safety. Part of Scale AI's RLHF pipeline.",
            requirements=["Strong analytical skills", "Domain expertise a plus"],
            responsibilities=["Evaluate LLM outputs", "Rank response quality", "Provide human feedback"],
            tags=["LLM", "RLHF", "AI Evaluator", "Scale AI", "Remotasks"]
        ),
        JobPosting(
            title="Coding Task Specialist",
            company="Scale AI + Remotasks",
            location="Remote",
            url="https://app.remotasks.com/#coding",
            source="Remotasks",
            compensation="$25-$50/hr",
            job_type="Freelance/Contract",
            description="Evaluate AI-generated code, write reference solutions, and assess coding task accuracy for Scale AI.",
            requirements=["4+ years software engineering", "Strong CS fundamentals"],
            responsibilities=["Evaluate AI code", "Write reference solutions", "Assess accuracy"],
            tags=["Coding", "AI Trainer", "Scale AI", "Remotasks"]
        ),

        # =====================================================
        # Babel Audio (156)
        # =====================================================
        JobPosting(
            title="AI Trainer - English Dialogue & Speech",
            company="Babel Audio",
            location="Remote",
            url="https://www.indeed.com/cmp/Babel-Audio/jobs",
            source="Indeed",
            compensation="Varies",
            job_type="Contract",
            description="Record short emotional scripts and dialogue for AI speech model training. Work with major tech companies to collect audio data for next-gen AI models.",
            requirements=["Voice acting experience preferred", "Understanding of emotional nuance", "Ability to convey wide range of feelings vocally"],
            responsibilities=["Record emotional scripts", "Provide dialogue samples", "Support speech AI training"],
            tags=["Speech AI", "Voice Acting", "Audio Data", "AI Trainer", "Babel Audio"]
        ),


        # =====================================================
        # Odixcity Consulting (158)
        # =====================================================
        JobPosting(
            title="LLM Trainer / AI Trainer",
            company="Odixcity Consulting",
            location="Remote",
            url="https://www.indeed.com/cmp/Odixcity-Consulting/jobs",
            source="Indeed",
            compensation="Varies",
            job_type="Contract",
            description="LLM training and AI data annotation role. Train and evaluate large language models through annotation and feedback.",
            requirements=["Strong analytical skills", "Familiarity with AI/ML concepts a plus"],
            responsibilities=["Train LLMs through annotation", "Evaluate model outputs", "Provide quality feedback"],
            tags=["LLM Trainer", "AI Trainer", "Data Annotation", "Odixcity Consulting"]
        ),

        # =====================================================
        # Taskify AI (159-161)
        # =====================================================
        JobPosting(
            title="Data Annotator",
            company="Taskify AI",
            location="Remote",
            url="https://www.indeed.com/cmp/Taskify-Ai/jobs#annotator",
            source="Indeed",
            compensation="Varies",
            job_type="Contract",
            description="Data annotation for AI training datasets. Label text, images, and other data types following annotation guidelines.",
            requirements=["Attention to detail", "Ability to follow guidelines"],
            responsibilities=["Label and annotate data", "Maintain quality standards", "Follow annotation protocols"],
            tags=["Data Annotator", "AI Training", "Taskify AI"]
        ),
        JobPosting(
            title="Data Annotation Specialist",
            company="Taskify AI",
            location="Remote",
            url="https://www.indeed.com/cmp/Taskify-Ai/jobs#specialist",
            source="Indeed",
            compensation="Varies",
            job_type="Contract",
            description="Specialized data annotation role requiring expertise in specific domains for high-quality AI training data creation.",
            requirements=["Domain expertise", "Strong attention to detail"],
            responsibilities=["Create high-quality annotations", "Quality assurance", "Support AI model training"],
            tags=["Data Annotation", "Specialist", "AI Training", "Taskify AI"]
        ),
        JobPosting(
            title="Digital Annotation Expert",
            company="Taskify AI",
            location="Remote",
            url="https://www.indeed.com/cmp/Taskify-Ai/jobs#digital",
            source="Indeed",
            compensation="Varies",
            job_type="Contract",
            description="Expert-level digital annotation for AI/ML model training. Handle complex annotation tasks across multiple media types.",
            requirements=["Expert annotation experience", "Technical aptitude"],
            responsibilities=["Complex annotation tasks", "Multi-media labeling", "Quality control"],
            tags=["Digital Annotation", "Expert", "AI Training", "Taskify AI"]
        ),

        # =====================================================
        # Neon (162)
        # =====================================================
        JobPosting(
            title="AI Trainers / Audio Data Labelers",
            company="Neon",
            location="Remote",
            url="https://www.indeed.com/cmp/Neon/jobs#ai-trainer",
            source="Indeed",
            compensation="Varies",
            job_type="Contract",
            description="AI training and audio data labeling for Neon. Label and annotate audio datasets for speech and language model training.",
            requirements=["Good hearing", "Attention to detail", "Language proficiency"],
            responsibilities=["Label audio data", "Annotate speech patterns", "Support audio AI training"],
            tags=["Audio Data", "AI Trainer", "Data Labeling", "Neon"]
        ),

        # =====================================================
        # NVIDIA via Sustainable Talent (staffing agency) (163)
        # =====================================================
        JobPosting(
            title="Generative AI Annotation Operations Engineer",
            company="NVIDIA via Sustainable Talent",
            location="Remote (US, Pacific Time preferred)",
            url="https://www.indeed.com/viewjob?jk=b6200c984d4f30ff",
            source="Indeed",
            compensation="$40-$60/hr W-2",
            job_type="Full-Time Contract (W-2)",
            description="Design and optimize annotation workflows that support NVIDIA foundational model training and evaluation. Work across teams to automate pipelines, configure tools, and support multi-stage, model-in-the-loop workflows essential to LLM development. Sustainable Talent is the staffing agency recruiting on behalf of NVIDIA.",
            requirements=["2-5 years in data annotation operations, ML data workflows, or data engineering", "Python scripting proficiency", "Experience with annotation tooling (Scale AI, Labelbox, SuperAnnotate preferred)", "AWS S3 and cloud storage pipelines"],
            responsibilities=["Design annotation workflows for model training", "Automate data pipelines (JSON, JSONL, CSV)", "Configure annotation tools", "Support multi-stage model-in-the-loop workflows"],
            tags=["NVIDIA", "GenAI", "Annotation Operations", "ML Workflows", "Sustainable Talent", "Staffing"]
        ),
        JobPosting(
            title="Data Annotation Engineer",
            company="NVIDIA via Sustainable Talent",
            location="Remote (US, Pacific Time preferred)",
            url="https://job-boards.greenhouse.io/sustainabletalent/jobs/4590677005",
            source="Sustainable Talent (Greenhouse)",
            compensation="$40-$60/hr W-2",
            job_type="Full-Time Contract (W-2)",
            description="Support annotation operations across varied data types for NVIDIA's foundational model training. Focus on data quality, pipeline engineering, and annotation workflow management. Sustainable Talent is the staffing agency recruiting on behalf of NVIDIA.",
            requirements=["2-5 years in data annotation operations, ML data workflows, or data engineering", "Python scripting proficiency", "Experience with annotation tooling preferred"],
            responsibilities=["Support annotation operations across data types", "Ensure data quality for model training", "Build and maintain data pipelines", "Collaborate with ML teams"],
            tags=["NVIDIA", "Data Annotation", "Engineering", "ML Workflows", "Sustainable Talent", "Staffing"]
        ),
        JobPosting(
            title="Machine Learning Engineer - Data Annotation Platforms",
            company="NVIDIA via Sustainable Talent",
            location="Santa Clara, CA (Hybrid/Remote)",
            url="https://www.ziprecruiter.com/c/Sustainable-Talent/Job/Machine-Learning-Engineer/-in-Santa-Clara,CA",
            source="ZipRecruiter",
            compensation="$60-$95/hr W-2",
            job_type="Full-Time Contract (W-2)",
            description="Work alongside NVIDIA's data annotation platform team, addressing evolving annotation data needs for training and evaluating LLMs. Requires demonstrated skills with large language models, NLP, and multi-modal models. Sustainable Talent is the staffing agency recruiting on behalf of NVIDIA.",
            requirements=["Experience with LLMs, NLP, and multi-modal models", "Strong ML engineering background", "Python proficiency", "Experience with annotation platforms"],
            responsibilities=["Build ML solutions for annotation platforms", "Address annotation data needs for LLM training/evaluation", "Collaborate with data annotation platform team", "Develop and optimize data pipelines"],
            tags=["NVIDIA", "Machine Learning", "Annotation Platforms", "LLM", "NLP", "Sustainable Talent", "Staffing"]
        ),
        JobPosting(
            title="AI Safety Engineer",
            company="NVIDIA via Sustainable Talent",
            location="Santa Clara, CA (Hybrid/Remote)",
            url="https://www.ziprecruiter.com/c/Sustainable-Talent/Job/AI-Safety-Engineer/-in-Santa-Clara,CA?jid=fa9686f87a770969",
            source="ZipRecruiter",
            compensation="$90-$130/hr W-2",
            job_type="Full-Time Contract (W-2)",
            description="Contribute to safety repositories and develop safety tools to help ML teams be more effective at NVIDIA. Conduct data pre-processing and analysis, perform exploratory data analysis, and collaborate with multidisciplinary teams. Experience with multimodal and/or multilingual Content Safety, legal and regulatory compliance sought. Sustainable Talent is the staffing agency recruiting on behalf of NVIDIA.",
            requirements=["Bachelor's or Master's in CS or related field", "2+ years as ML Engineer or Deep Learning Scientist", "Strong Python skills", "Track record of delivering ML solutions"],
            responsibilities=["Develop AI safety tools and repositories", "Data pre-processing and analysis", "Exploratory data analysis for model performance", "Collaborate with product engineers, data scientists, and analysts"],
            tags=["NVIDIA", "AI Safety", "ML Engineering", "Content Safety", "Sustainable Talent", "Staffing"]
        ),

        # =====================================================
        # DataAnnotation.tech - Additional Indeed-specific roles (164-167)
        # =====================================================
        JobPosting(
            title="Software Developer - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.dataannotation.tech/#software-dev",
            source="Indeed / DataAnnotation.tech",
            compensation="$25-$50/hr",
            job_type="Freelance/Contract",
            description="Train AI models on software development. Write, review, and evaluate code across multiple programming languages.",
            requirements=["4+ years software development", "Proficiency in multiple languages"],
            responsibilities=["Train AI on coding tasks", "Evaluate code generation", "Write reference solutions"],
            tags=["Software Developer", "AI Trainer", "Coding", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="API Developer - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.dataannotation.tech/#api-dev",
            source="Indeed / DataAnnotation.tech",
            compensation="$25-$50/hr",
            job_type="Freelance/Contract",
            description="Train AI models on API design, RESTful services, integration patterns, and backend development.",
            requirements=["API development experience", "REST/GraphQL expertise"],
            responsibilities=["Train AI on API concepts", "Evaluate API-related code", "Provide technical feedback"],
            tags=["API Developer", "AI Trainer", "Backend", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Video Editor - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.dataannotation.tech/#video-editor",
            source="Indeed / DataAnnotation.tech",
            compensation="$15-$35/hr",
            job_type="Freelance/Contract",
            description="Train AI models on video editing concepts, workflows, and creative decisions.",
            requirements=["Video editing experience", "Familiarity with editing software"],
            responsibilities=["Train AI on video editing", "Evaluate creative outputs", "Provide feedback"],
            tags=["Video Editor", "AI Trainer", "Creative", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Primary Teacher - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.dataannotation.tech/#teacher",
            source="Indeed / DataAnnotation.tech",
            compensation="$15-$35/hr",
            job_type="Freelance/Contract",
            description="Train AI models on K-12 education concepts, pedagogy, and age-appropriate communication.",
            requirements=["Teaching experience", "Education background"],
            responsibilities=["Train AI on education topics", "Evaluate pedagogical accuracy", "Provide feedback"],
            tags=["Teacher", "Education", "AI Trainer", "Surge AI + DataAnnotation"]
        ),

        # DataAnnotation — Additional Indeed domain-specific AI Trainer roles
        JobPosting(
            title="Registered Nurse - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=5e20b2704a06b6f0",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on nursing, patient care, clinical assessment, pharmacology, care planning, nursing ethics, and evidence-based practice.",
            requirements=["Active RN license", "Clinical nursing experience"],
            responsibilities=["Train AI on clinical nursing", "Evaluate medical accuracy", "Provide healthcare feedback"],
            tags=["Nursing", "Healthcare", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Clinical Researcher - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=e968c0b5dfb8ba08",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on clinical research methodology, study design, data analysis, and evidence-based medicine.",
            requirements=["Clinical research experience", "Advanced degree preferred"],
            responsibilities=["Train AI on clinical research", "Evaluate scientific reasoning", "Provide expert feedback"],
            tags=["Clinical Research", "Healthcare", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Biology Research Scientist - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=f51df7b8799ab664",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on biology research, experimental design, molecular biology, genetics, and scientific methodology.",
            requirements=["Biology degree", "Research experience"],
            responsibilities=["Train AI on biology", "Evaluate scientific accuracy", "Provide expert feedback"],
            tags=["Biology", "Research", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Chemistry Expert - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=0ad559761dbafdfe",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Evaluate and improve AI understanding of chemical principles. Requires advanced degree (PhD or MS) in Chemistry with 3+ years experience.",
            requirements=["PhD or MS in Chemistry", "3+ years research/teaching/applied chemistry"],
            responsibilities=["Evaluate chemistry content", "Generate high-quality chemistry training data", "Provide expert feedback"],
            tags=["Chemistry", "STEM", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="AI Training Physics - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=26d36994f8b2bd0e",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on physics concepts, problem-solving, and scientific reasoning. Master's/PhD preferred.",
            requirements=["Advanced physics degree preferred", "Expert-level physics understanding"],
            responsibilities=["Train AI on physics", "Evaluate scientific reasoning", "Provide expert feedback"],
            tags=["Physics", "STEM", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Biostatistician - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=be183e7a31834682",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on biostatistics, clinical trial design, statistical analysis, and epidemiological methods.",
            requirements=["Biostatistics degree", "Clinical data analysis experience"],
            responsibilities=["Train AI on biostatistics", "Evaluate statistical reasoning", "Provide expert feedback"],
            tags=["Biostatistics", "STEM", "Healthcare", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Data Scientist - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=e63d8be4ffa083da",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on data science concepts, ML, statistical modeling, and analytical methods.",
            requirements=["Data science background", "ML and statistics expertise"],
            responsibilities=["Train AI on data science", "Evaluate analytical reasoning", "Provide expert feedback"],
            tags=["Data Science", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Machine Learning Engineer - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=90f90a0091462141",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on ML engineering, model architecture, training pipelines, and deployment.",
            requirements=["ML engineering experience", "Strong Python and framework knowledge"],
            responsibilities=["Train AI on ML concepts", "Evaluate model-related reasoning", "Provide expert feedback"],
            tags=["Machine Learning", "Engineering", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Software Engineer - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=b4a579230706c1cf",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Part of DataAnnotation's 100K+ coding community. Train AI on software engineering, debugging, and system design.",
            requirements=["2+ years software engineering", "Proficiency in major languages"],
            responsibilities=["Evaluate AI-generated code", "Write reference solutions", "Provide coding feedback"],
            tags=["Software Engineering", "Coding", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Security Engineer - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=f036bc08e7f52e37",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on cybersecurity concepts, threat analysis, secure coding, and network security. 2+ years cybersecurity experience required.",
            requirements=["2+ years cybersecurity experience", "Security domain expertise"],
            responsibilities=["Train AI on security", "Evaluate security-related reasoning", "Provide expert feedback"],
            tags=["Cybersecurity", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="API Test Engineer - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=3c88ec8a06bd0f4d",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on API testing, quality assurance, test automation, and integration testing.",
            requirements=["API testing experience", "QA/test automation skills"],
            responsibilities=["Train AI on API testing", "Evaluate QA reasoning", "Provide expert feedback"],
            tags=["API Testing", "QA", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Actuary - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=864296ff6da27c6a",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on actuarial science, risk modeling, probability, and insurance mathematics.",
            requirements=["Actuarial credentials or exam progress", "Statistical modeling expertise"],
            responsibilities=["Train AI on actuarial concepts", "Evaluate risk modeling", "Provide expert feedback"],
            tags=["Actuary", "Finance", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Personal Injury Paralegal - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=6d29d28da5409043",
            source="Indeed",
            compensation="$20-$40/hr",
            job_type="Freelance/Contract",
            description="Train AI on personal injury law, case management, legal research, and documentation.",
            requirements=["Paralegal certification or experience", "Personal injury law knowledge"],
            responsibilities=["Train AI on legal concepts", "Evaluate legal reasoning", "Provide expert feedback"],
            tags=["Paralegal", "Legal", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Quantitative Researcher - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=0faa5a2d437437fb",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on quantitative research, statistical methods, and expert-level scientific reasoning.",
            requirements=["Quantitative research background", "Advanced degree preferred"],
            responsibilities=["Train AI on quantitative methods", "Evaluate analytical accuracy", "Provide expert feedback"],
            tags=["Quantitative Research", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Freelance Copywriter - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=a602bda4317ee37b",
            source="Indeed",
            compensation="$20-$40/hr",
            job_type="Freelance/Contract",
            description="Train AI on copywriting, creative writing, content strategy. Experience in literature, philosophy, theology, or editorial work valued.",
            requirements=["Writing/editorial experience", "Content strategy background preferred"],
            responsibilities=["Train AI on writing quality", "Evaluate creative output", "Provide editorial feedback"],
            tags=["Copywriting", "Writing", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Digital Marketer - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=246c7b9998a77b13",
            source="Indeed",
            compensation="$20-$40/hr",
            job_type="Freelance/Contract",
            description="Train AI on digital marketing, SEO, social media strategy, content marketing, and analytics.",
            requirements=["Digital marketing experience", "SEO/SEM knowledge"],
            responsibilities=["Train AI on marketing", "Evaluate marketing reasoning", "Provide expert feedback"],
            tags=["Digital Marketing", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Social Media Manager - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=2597061277bad8a5",
            source="Indeed",
            compensation="$20-$40/hr",
            job_type="Freelance/Contract",
            description="Train AI on social media management, community engagement, and platform-specific strategies.",
            requirements=["Social media management experience"],
            responsibilities=["Train AI on social media", "Evaluate content strategy", "Provide expert feedback"],
            tags=["Social Media", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Communications Manager - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=1e02997983e84f48",
            source="Indeed",
            compensation="$20-$40/hr",
            job_type="Freelance/Contract",
            description="Train AI on communications, PR, corporate messaging, and stakeholder engagement.",
            requirements=["Communications/PR experience"],
            responsibilities=["Train AI on communications", "Evaluate messaging quality", "Provide expert feedback"],
            tags=["Communications", "PR", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="ESL Tutor - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=44c45499874d65c0",
            source="Indeed",
            compensation="$20-$35/hr",
            job_type="Freelance/Contract",
            description="Train AI on ESL pedagogy, language acquisition, grammar instruction, and cross-cultural communication.",
            requirements=["ESL teaching experience", "TESOL/TEFL certification preferred"],
            responsibilities=["Train AI on ESL concepts", "Evaluate language teaching quality", "Provide expert feedback"],
            tags=["ESL", "Education", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="High School Teacher - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=5d9fb97530488903",
            source="Indeed",
            compensation="$20-$35/hr",
            job_type="Freelance/Contract",
            description="Train AI on high school curriculum, pedagogy, and subject-matter expertise across disciplines.",
            requirements=["Teaching certification or experience", "Subject matter expertise"],
            responsibilities=["Train AI on education topics", "Evaluate pedagogical accuracy", "Provide feedback"],
            tags=["Education", "Teaching", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Proofreader - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=263beba686b8712a",
            source="Indeed",
            compensation="$20-$35/hr",
            job_type="Freelance/Contract",
            description="Train AI on proofreading, grammar, style consistency, and editorial standards.",
            requirements=["Proofreading/editing experience", "Attention to detail"],
            responsibilities=["Train AI on editing quality", "Evaluate language accuracy", "Provide editorial feedback"],
            tags=["Proofreading", "Writing", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Associate Editor - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=40f9b480a774a9ab",
            source="Indeed",
            compensation="$20-$35/hr",
            job_type="Freelance/Contract",
            description="Train AI on editorial processes, content curation, fact-checking, and publishing standards.",
            requirements=["Editorial experience", "Content management skills"],
            responsibilities=["Train AI on editorial quality", "Evaluate content accuracy", "Provide editorial feedback"],
            tags=["Editor", "Publishing", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Customer Success Manager - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=bff51f2b8ab3858c",
            source="Indeed",
            compensation="$20-$40/hr",
            job_type="Freelance/Contract",
            description="Train AI on customer success, account management, retention strategies, and client communication.",
            requirements=["Customer success/account management experience"],
            responsibilities=["Train AI on customer success", "Evaluate business reasoning", "Provide expert feedback"],
            tags=["Customer Success", "Business", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Digital Content Editor - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=f726301a072c3334",
            source="Indeed",
            compensation="$20-$35/hr",
            job_type="Freelance/Contract",
            description="Train AI on digital content editing, web publishing, content strategy, and multimedia production.",
            requirements=["Digital content experience", "Editorial skills"],
            responsibilities=["Train AI on content editing", "Evaluate digital content quality", "Provide feedback"],
            tags=["Digital Content", "Editor", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Creative Developer - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=735ed79802df697c",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on creative development, interactive design, front-end technologies, and creative coding.",
            requirements=["Creative development experience", "Front-end/interactive skills"],
            responsibilities=["Train AI on creative dev", "Evaluate creative code", "Provide expert feedback"],
            tags=["Creative Development", "Coding", "AI Trainer", "Surge AI + DataAnnotation"]
        ),
        JobPosting(
            title="Psychometrician - AI Trainer",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.indeed.com/viewjob?jk=psychometrician-da",
            source="Indeed",
            compensation="$40+/hr",
            job_type="Freelance/Contract",
            description="Train AI on psychometrics, test design, measurement theory, and assessment validity.",
            requirements=["Psychometrics expertise", "Advanced degree preferred"],
            responsibilities=["Train AI on psychometrics", "Evaluate measurement reasoning", "Provide expert feedback"],
            tags=["Psychometrics", "Assessment", "AI Trainer", "Surge AI + DataAnnotation"]
        ),

        # =====================================================
        # OpenTrain AI (168)
        # =====================================================
        JobPosting(
            title="AI Trainer / Data Labeler",
            company="OpenTrain AI",
            location="Remote",
            url="https://www.opentrain.ai/",
            source="OpenTrain AI",
            compensation="Varies by project",
            job_type="Freelance/Contract",
            description="Platform connecting AI trainers and data labelers with companies needing human feedback and annotation for AI model development.",
            requirements=["Computer literacy", "Domain expertise a plus"],
            responsibilities=["Label and annotate data", "Provide human feedback", "Train AI models"],
            tags=["AI Trainer", "Data Labeler", "OpenTrain AI"]
        ),

        # =====================================================
        # Comrise (169)
        # =====================================================
        JobPosting(
            title="AI Data Annotation Specialist",
            company="Comrise",
            location="Remote",
            url="https://www.indeed.com/cmp/Comrise/jobs#ai-annotation",
            source="Indeed",
            compensation="$18-$35/hr",
            job_type="Contract",
            description="AI data annotation role at Comrise supporting LLM training. Annotate and label datasets for machine learning model development.",
            requirements=["Attention to detail", "Data annotation experience preferred"],
            responsibilities=["Annotate training data", "Quality assurance", "Follow labeling guidelines"],
            tags=["Data Annotation", "AI Training", "Comrise"]
        ),

        # =====================================================
        # Labelbox + Alignerr — Expanded Roles from alignerr.com/jobs (170-210)
        # =====================================================

        # --- Domain/STEM Specialist Roles ---
        JobPosting(
            title="AI Trainer for Physics",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4446933007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI models on physics concepts, problem-solving, and scientific reasoning.",
            requirements=["Advanced physics degree", "Strong problem-solving skills"],
            responsibilities=["Train AI on physics", "Evaluate scientific accuracy", "Provide expert feedback"],
            tags=["Physics", "STEM", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Chemistry",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4446860007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI models on general chemistry, chemical reactions, and analytical methods.",
            requirements=["Chemistry degree or equivalent expertise"],
            responsibilities=["Train AI on chemistry", "Evaluate chemical reasoning", "Provide expert feedback"],
            tags=["Chemistry", "STEM", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Organic Chemistry",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4603786007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on organic chemistry, reaction mechanisms, synthesis planning, and stereochemistry.",
            requirements=["Advanced organic chemistry expertise"],
            responsibilities=["Train AI on organic chemistry", "Evaluate synthesis reasoning", "Provide expert feedback"],
            tags=["Organic Chemistry", "STEM", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Biochemistry",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4603800007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on biochemistry, molecular biology, enzymology, and metabolic pathways.",
            requirements=["Advanced biochemistry degree"],
            responsibilities=["Train AI on biochemistry", "Evaluate molecular reasoning", "Provide expert feedback"],
            tags=["Biochemistry", "STEM", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Biology",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4463599007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI models on biology concepts across ecology, genetics, evolution, and cell biology.",
            requirements=["Biology degree or equivalent"],
            responsibilities=["Train AI on biology", "Evaluate biological accuracy", "Provide expert feedback"],
            tags=["Biology", "STEM", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Molecular Biology",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4520448007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on molecular biology, gene expression, protein structure, and genomics.",
            requirements=["Advanced molecular biology expertise"],
            responsibilities=["Train AI on molecular biology", "Evaluate genomics reasoning", "Provide expert feedback"],
            tags=["Molecular Biology", "STEM", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Machine Learning",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4520457007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI models on ML concepts, neural networks, deep learning, and model evaluation.",
            requirements=["ML/AI expertise", "Advanced degree preferred"],
            responsibilities=["Train AI on ML concepts", "Evaluate technical reasoning", "Provide expert feedback"],
            tags=["Machine Learning", "AI", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Medicine",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://startup.jobs/ai-trainer-for-medicine-freelance-remote-alignerr-6273331",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on medical knowledge, clinical reasoning, diagnostics, and treatment protocols.",
            requirements=["Medical degree or advanced clinical expertise"],
            responsibilities=["Train AI on medical topics", "Evaluate clinical reasoning", "Provide expert feedback"],
            tags=["Medicine", "Healthcare", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for PhD Expertise",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4600287007",
            source="Alignerr (Labelbox) / Greenhouse",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Leverage PhD-level expertise across any discipline to train AI. Tasks include step-by-step solutions, analyzing data, interpreting frameworks, and pushing model boundaries.",
            requirements=["PhD in any field", "Deep domain expertise"],
            responsibilities=["Provide expert solutions", "Identify AI biases and limitations", "Conduct adversarial tests"],
            tags=["PhD", "Research", "AI Trainer", "Labelbox", "Alignerr"]
        ),

        # --- Coding & Technical Roles ---
        JobPosting(
            title="Coders - AI Training",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4362423007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on coding. Requires proficiency in Python, Java, JavaScript/TypeScript, SQL, C/C++/C#, and/or HTML.",
            requirements=["Bachelor's in CS or equivalent", "Proficiency in major programming languages"],
            responsibilities=["Evaluate AI-generated code", "Write reference solutions", "Provide coding feedback"],
            tags=["Coding", "Software Engineering", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer, SOQL Developer",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4519511007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI models on Salesforce SOQL query language and Salesforce platform development.",
            requirements=["SOQL expertise", "Salesforce development experience"],
            responsibilities=["Train AI on SOQL", "Evaluate query accuracy", "Provide Salesforce feedback"],
            tags=["SOQL", "Salesforce", "AI Trainer", "Labelbox", "Alignerr"]
        ),

        # --- Generalist & Writing Roles ---
        JobPosting(
            title="AI Training for Generalist",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4362432007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="General AI training role covering broad topics. No specialized domain required.",
            requirements=["Strong analytical skills", "Clear communication"],
            responsibilities=["Evaluate AI responses", "Provide general feedback", "Train on diverse topics"],
            tags=["Generalist", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="Generalist - Writing (English)",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/27462526-34cb-42cf-a6ce-00e3af846b50",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="General writing and evaluation role in English for AI model training.",
            requirements=["Native-level English", "Strong writing skills"],
            responsibilities=["Evaluate AI writing", "Provide editorial feedback", "Train on writing quality"],
            tags=["Writing", "English", "Generalist", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Creative Writing",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4604120007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on creative writing including fiction, poetry, screenwriting, and narrative craft.",
            requirements=["Creative writing portfolio", "MFA or equivalent experience preferred"],
            responsibilities=["Train AI on creative writing", "Evaluate narrative quality", "Provide creative feedback"],
            tags=["Creative Writing", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Training for Technical Writers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4362428007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on technical writing including documentation, API docs, and technical communication.",
            requirements=["Technical writing experience", "Domain knowledge"],
            responsibilities=["Evaluate technical documentation", "Train on clarity and accuracy", "Provide writing feedback"],
            tags=["Technical Writing", "AI Trainer", "Labelbox", "Alignerr"]
        ),

        # --- Domain-Specific Roles ---
        JobPosting(
            title="AI Training for Data Science",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4410875007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on data science concepts, statistical analysis, and data engineering.",
            requirements=["Data science background", "Statistical modeling expertise"],
            responsibilities=["Train AI on data science", "Evaluate analytical reasoning", "Provide expert feedback"],
            tags=["Data Science", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Training for Mathematics",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4410921007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on mathematics including algebra, calculus, statistics, and advanced topics.",
            requirements=["Mathematics degree or equivalent", "Strong problem-solving"],
            responsibilities=["Train AI on math", "Evaluate mathematical reasoning", "Write math problems"],
            tags=["Mathematics", "STEM", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Training for Financial Analyst",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4409178007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on financial analysis, modeling, valuation, and investment concepts.",
            requirements=["Finance background", "Financial modeling experience"],
            responsibilities=["Train AI on finance", "Evaluate financial reasoning", "Provide expert feedback"],
            tags=["Finance", "Financial Analysis", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Training for K-12 Education Expert",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4372465007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on K-12 education, pedagogy, curriculum development, and age-appropriate content.",
            requirements=["Teaching experience", "Education background"],
            responsibilities=["Train AI on education", "Evaluate pedagogical accuracy", "Provide feedback"],
            tags=["K-12", "Education", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Training for Energy and Power",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4410875007#energy",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Train AI on energy systems, power engineering, renewable energy, and grid technologies.",
            requirements=["Energy/power engineering background"],
            responsibilities=["Train AI on energy topics", "Evaluate technical accuracy", "Provide expert feedback"],
            tags=["Energy", "Power", "Engineering", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="Research Analyst",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4468e480-1e76-4542-85a1-1088d99e4525",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Research analyst role supporting AI training through structured research and analysis.",
            requirements=["Research experience", "Analytical skills"],
            responsibilities=["Conduct research for AI training", "Analyze data", "Provide structured insights"],
            tags=["Research", "Analyst", "AI Trainer", "Labelbox", "Alignerr"]
        ),

        # --- Voice & Audio Roles ---
        JobPosting(
            title="AI Voice Trainer - English (Contract)",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/acc2919d-4649-43cb-beb7-4cc70d1d814a",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Contract",
            description="Record expressive, high-quality audio samples for AI voice systems (Project Rainforest).",
            requirements=["Clear voice", "Quiet recording space", "Ability to follow recording guidelines"],
            responsibilities=["Record audio samples", "Provide expressive voice data", "Follow quality standards"],
            tags=["Voice", "Audio", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="Voice Actor - French Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/3bc4f9b3-6144-44cf-97ab-4d41206e3680",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Contract",
            description="French-speaking voice actor for AI voice training data collection.",
            requirements=["Native French speaker", "Voice acting skills"],
            responsibilities=["Record French audio", "Provide expressive voice data", "Follow quality standards"],
            tags=["Voice Actor", "French", "Audio", "AI Trainer", "Labelbox", "Alignerr"]
        ),

        # --- Language-Specific AI Trainer Roles ---
        JobPosting(
            title="AI Trainer for English Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4458830007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native-level English proficiency required. Evaluate and train AI on English language generation.",
            requirements=["Native English proficiency", "Excellent spelling and grammar"],
            responsibilities=["Train AI on English", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["English", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Spanish Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4366849007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Spanish proficiency required for AI language training.",
            requirements=["Native Spanish proficiency", "English fluency"],
            responsibilities=["Train AI on Spanish", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Spanish", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for French Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4382721007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native French proficiency required for AI language training.",
            requirements=["Native French proficiency", "English fluency"],
            responsibilities=["Train AI on French", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["French", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for German Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4372448007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native German proficiency required for AI language training.",
            requirements=["Native German proficiency", "English fluency"],
            responsibilities=["Train AI on German", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["German", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Italian Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4382725007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Italian proficiency required for AI language training.",
            requirements=["Native Italian proficiency", "English fluency"],
            responsibilities=["Train AI on Italian", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Italian", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Korean Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4382236007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Korean proficiency required for AI language training.",
            requirements=["Native Korean proficiency", "English fluency"],
            responsibilities=["Train AI on Korean", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Korean", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Japanese Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4382711007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Japanese proficiency required for AI language training.",
            requirements=["Native Japanese proficiency", "English fluency"],
            responsibilities=["Train AI on Japanese", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Japanese", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Mandarin Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4464813007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Mandarin proficiency required for AI language training.",
            requirements=["Native Mandarin proficiency", "English fluency"],
            responsibilities=["Train AI on Mandarin", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Mandarin", "Chinese", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Arabic Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4446544007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Arabic proficiency required for AI language training.",
            requirements=["Native Arabic proficiency", "English fluency"],
            responsibilities=["Train AI on Arabic", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Arabic", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Hindi Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4373626007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Hindi proficiency required for AI language training.",
            requirements=["Native Hindi proficiency", "English fluency"],
            responsibilities=["Train AI on Hindi", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Hindi", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Vietnamese Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4366840007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Vietnamese proficiency required for AI language training.",
            requirements=["Native Vietnamese proficiency", "English fluency"],
            responsibilities=["Train AI on Vietnamese", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Vietnamese", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Indonesian Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4382713007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Indonesian proficiency required for AI language training.",
            requirements=["Native Indonesian proficiency", "English fluency"],
            responsibilities=["Train AI on Indonesian", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Indonesian", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Thai Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4514360007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Thai proficiency required for AI language training.",
            requirements=["Native Thai proficiency", "English fluency"],
            responsibilities=["Train AI on Thai", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Thai", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Filipino (Tagalog) Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4514364007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Filipino/Tagalog proficiency required for AI language training.",
            requirements=["Native Filipino/Tagalog proficiency", "English fluency"],
            responsibilities=["Train AI on Filipino", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Filipino", "Tagalog", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Portuguese (Brazil) Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4366849007#portuguese",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Brazilian Portuguese proficiency required for AI language training.",
            requirements=["Native Brazilian Portuguese proficiency", "English fluency"],
            responsibilities=["Train AI on Portuguese", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Portuguese", "Brazilian", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Russian Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4464826007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Russian proficiency required for AI language training.",
            requirements=["Native Russian proficiency", "English fluency"],
            responsibilities=["Train AI on Russian", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Russian", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Dutch Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4464878007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Dutch proficiency required for AI language training.",
            requirements=["Native Dutch proficiency", "English fluency"],
            responsibilities=["Train AI on Dutch", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Dutch", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Romanian Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4464900007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Romanian proficiency required for AI language training.",
            requirements=["Native Romanian proficiency", "English fluency"],
            responsibilities=["Train AI on Romanian", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Romanian", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Bengali Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/4514363007",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Bengali proficiency required for AI language training.",
            requirements=["Native Bengali proficiency", "English fluency"],
            responsibilities=["Train AI on Bengali", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Bengali", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Trainer for Hausa Writers/Speakers",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://startup.jobs/ai-trainer-for-hausa-writers-speakers-freelance-remote-alignerr-6273741",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Native Hausa proficiency required for AI language training.",
            requirements=["Native Hausa proficiency", "English fluency"],
            responsibilities=["Train AI on Hausa", "Evaluate language quality", "Provide linguistic feedback"],
            tags=["Hausa", "Language", "AI Trainer", "Labelbox", "Alignerr"]
        ),
        JobPosting(
            title="AI Language Expert - Korean",
            company="Labelbox + Alignerr",
            location="Remote",
            url="https://www.alignerr.com/jobs/27ee899f-8851-48d3-86ea-df75c6b8f4d7",
            source="Alignerr (Labelbox)",
            compensation="Up to $150/hr",
            job_type="Freelance/Contract",
            description="Korean language expert for advanced AI language model training and evaluation.",
            requirements=["Native Korean proficiency", "Linguistic expertise"],
            responsibilities=["Expert-level AI language training", "Evaluate nuanced outputs", "Provide linguistic feedback"],
            tags=["Korean", "Language Expert", "AI Trainer", "Labelbox", "Alignerr"]
        ),

        # ============================================================
        # Harvey AI — Legal AI (builds AI for law firms)
        # ============================================================
        JobPosting(
            title="Legal Engineer",
            company="Harvey AI",
            location="San Francisco, CA / New York, NY",
            url="https://jobs.ashbyhq.com/harvey/3fc0953f",
            source="Harvey AI Careers (Ashby)",
            compensation="$200K-$300K base + equity",
            job_type="Full-Time",
            description="Build and evaluate AI systems for legal workflows. Requires deep legal domain expertise to train and validate AI models for legal reasoning, contract analysis, and regulatory compliance.",
            requirements=["JD from top-tier law school", "3+ years at top law firm", "Strong analytical and writing skills"],
            responsibilities=["Train AI on legal reasoning", "Evaluate legal AI outputs", "Design legal benchmarks", "Collaborate with ML engineers"],
            tags=["Legal", "AI Training", "RLHF", "Legal AI", "Harvey AI", "Domain Expert"],
            job_category="Corporate/In-House"
        ),
        JobPosting(
            title="Legal Engineer - Product Specialist, Tax",
            company="Harvey AI",
            location="San Francisco, CA / New York, NY",
            url="https://jobs.ashbyhq.com/harvey/11524077",
            source="Harvey AI Careers (Ashby)",
            compensation="$200K-$300K base + equity",
            job_type="Full-Time",
            description="Tax law specialist building and evaluating AI products for tax advisory, compliance, and planning workflows.",
            requirements=["JD or LLM in Tax", "3+ years tax practice at top firm", "Deep knowledge of federal/state tax code"],
            responsibilities=["Train AI on tax legal reasoning", "Evaluate tax AI outputs", "Design tax-specific benchmarks", "Build tax workflow products"],
            tags=["Legal", "Tax", "AI Training", "Legal AI", "Harvey AI", "Domain Expert"],
            job_category="Corporate/In-House"
        ),
        JobPosting(
            title="Legal Engineer, EMEA",
            company="Harvey AI",
            location="London, UK",
            url="https://jobs.ashbyhq.com/harvey/4154ddf4",
            source="Harvey AI Careers (Ashby)",
            compensation="Competitive (UK market)",
            job_type="Full-Time",
            description="Legal engineer focused on EMEA jurisdictions, training AI on UK/EU legal frameworks including GDPR, EU regulations, and common law traditions.",
            requirements=["Qualified solicitor/barrister (UK) or equivalent", "3+ years at top-tier firm", "EMEA legal expertise"],
            responsibilities=["Train AI on EMEA legal reasoning", "Evaluate cross-jurisdictional outputs", "Adapt US-trained models for EMEA markets"],
            tags=["Legal", "EMEA", "UK", "AI Training", "Legal AI", "Harvey AI", "Domain Expert"],
            job_category="Corporate/In-House"
        ),
        JobPosting(
            title="Applied Legal Researcher",
            company="Harvey AI",
            location="San Francisco, CA / New York, NY",
            url="https://jobs.ashbyhq.com/harvey/a35efd1c",
            source="Harvey AI Careers (Ashby)",
            compensation="$180K-$260K base + equity",
            job_type="Full-Time",
            description="Research role bridging legal domain expertise with AI capabilities. Design evaluation frameworks, create training data, and conduct legal reasoning research to improve AI model performance.",
            requirements=["JD or advanced legal degree", "Research experience", "Strong legal writing"],
            responsibilities=["Design legal evaluation benchmarks", "Create gold-standard training data", "Research legal reasoning improvements", "Publish findings"],
            tags=["Legal", "Research", "AI Training", "RLHF", "Legal AI", "Harvey AI", "Domain Expert"],
            job_category="Corporate/In-House"
        ),
        JobPosting(
            title="Legal Engineer - Custom Solutions",
            company="Harvey AI",
            location="San Francisco, CA / New York, NY",
            url="https://jobs.ashbyhq.com/harvey/f4248359",
            source="Harvey AI Careers (Ashby)",
            compensation="$200K-$300K base + equity",
            job_type="Full-Time",
            description="Build custom AI solutions for enterprise law firm clients. Tailor Harvey's AI platform to specific practice areas and client workflows.",
            requirements=["JD from top-tier law school", "3+ years at top firm", "Client-facing experience"],
            responsibilities=["Design custom legal AI workflows", "Train models on client-specific domains", "Evaluate output quality", "Manage enterprise deployments"],
            tags=["Legal", "Enterprise", "AI Training", "Legal AI", "Harvey AI", "Domain Expert"],
            job_category="Corporate/In-House"
        ),
        JobPosting(
            title="Legal Engineer, Dallas",
            company="Harvey AI",
            location="Dallas, TX",
            url="https://jobs.ashbyhq.com/harvey/3877a57f",
            source="Harvey AI Careers (Ashby)",
            compensation="$200K-$300K base + equity",
            job_type="Full-Time",
            description="Legal engineer based in Dallas supporting Texas/Southern US law firm clients. Train and evaluate AI for regional legal practices including energy, real estate, and corporate law.",
            requirements=["JD from top-tier law school", "3+ years at top firm", "Texas bar preferred"],
            responsibilities=["Train AI on legal reasoning", "Evaluate legal AI outputs", "Support regional clients", "Build legal benchmarks"],
            tags=["Legal", "AI Training", "Legal AI", "Harvey AI", "Domain Expert", "Texas"],
            job_category="Corporate/In-House"
        ),
        JobPosting(
            title="Legal Engineer - Product Specialist, Innovation",
            company="Harvey AI",
            location="San Francisco, CA / New York, NY",
            url="https://jobs.ashbyhq.com/harvey/9ec0be78",
            source="Harvey AI Careers (Ashby)",
            compensation="$200K-$300K base + equity",
            job_type="Full-Time",
            description="Innovation-focused legal engineer exploring new applications of AI in legal practice. Prototype new product features and evaluate novel AI capabilities for legal workflows.",
            requirements=["JD from top-tier law school", "3+ years at top firm", "Interest in legal tech innovation"],
            responsibilities=["Prototype new legal AI products", "Evaluate novel AI capabilities", "Design innovation benchmarks", "Cross-functional collaboration"],
            tags=["Legal", "Innovation", "AI Training", "Legal AI", "Harvey AI", "Domain Expert"],
            job_category="Corporate/In-House"
        ),

        # ============================================================
        # Healthcare / Medical AI Annotation Roles
        # ============================================================
        JobPosting(
            title="RN / Clinical Product Specialist - AI Validation",
            company="Hippocratic AI",
            location="Palo Alto, CA (Hybrid)",
            url="https://www.hippocraticai.com/careers#clinical",
            source="Hippocratic AI Careers",
            compensation="$120K-$180K base + equity",
            job_type="Full-Time",
            description="Registered Nurse validating AI healthcare agent outputs for clinical safety and accuracy. Hippocratic AI builds safety-focused LLMs for healthcare with $137M Series A.",
            requirements=["Active RN license", "3+ years clinical experience", "ICU/acute care preferred", "Familiarity with clinical decision support"],
            responsibilities=["Validate AI clinical outputs", "Design clinical safety benchmarks", "Review medical reasoning chains", "Provide expert clinical feedback"],
            tags=["Healthcare", "Medical", "Nursing", "RN", "AI Validation", "Clinical", "Domain Expert", "Hippocratic AI"],
            job_category="Corporate/In-House"
        ),
        JobPosting(
            title="AI Data Annotator - Biology / Cellular Imagery",
            company="Surge AI + DataAnnotation",
            location="Remote",
            url="https://www.dataannotation.tech/jobs#biology",
            source="DataAnnotation.tech (Surge AI)",
            compensation="$25-$50/hr",
            job_type="Freelance/Contract",
            description="Annotate biological and cellular imagery datasets for AI model training. Requires expertise in cell biology, histology, or microscopy image interpretation.",
            requirements=["Degree in Biology, Biochemistry, or related field", "Experience with cellular/microscopy imagery", "Attention to detail"],
            responsibilities=["Annotate biological imagery", "Label cellular structures", "Quality review annotations", "Provide domain feedback"],
            tags=["Healthcare", "Biology", "Medical", "Imagery", "Data Annotation", "Surge AI", "DataAnnotation"]
        ),
        JobPosting(
            title="Medical AI Trainer - Internal Medicine",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/invisibletech#medicine",
            source="Meridial (Invisible Technologies)",
            compensation="$40-$75/hr",
            job_type="Freelance/Contract",
            description="Train AI models on internal medicine knowledge including diagnosis, treatment protocols, and clinical reasoning. Expert-level medical annotation and RLHF.",
            requirements=["MD or DO degree", "Board-certified in Internal Medicine preferred", "Clinical experience"],
            responsibilities=["Train AI on medical reasoning", "Evaluate clinical AI outputs", "Annotate medical cases", "Provide expert feedback on diagnoses"],
            tags=["Healthcare", "Medical", "Internal Medicine", "AI Trainer", "RLHF", "Domain Expert", "Invisible Technologies", "Meridial"]
        ),
        JobPosting(
            title="Medical AI Trainer - Pharmacology",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/invisibletech#pharmacology",
            source="Meridial (Invisible Technologies)",
            compensation="$40-$75/hr",
            job_type="Freelance/Contract",
            description="Pharmacology expert training AI on drug interactions, dosing, side effects, and pharmaceutical knowledge for healthcare AI applications.",
            requirements=["PharmD, PhD in Pharmacology, or equivalent", "Clinical pharmacy experience preferred"],
            responsibilities=["Train AI on pharmacology", "Evaluate drug interaction outputs", "Annotate pharmaceutical data", "Review dosing recommendations"],
            tags=["Healthcare", "Medical", "Pharmacology", "AI Trainer", "RLHF", "Domain Expert", "Invisible Technologies", "Meridial"]
        ),
        JobPosting(
            title="Medical AI Trainer - Pathology",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/invisibletech#pathology",
            source="Meridial (Invisible Technologies)",
            compensation="$40-$75/hr",
            job_type="Freelance/Contract",
            description="Pathology specialist training AI on histopathology, cytology, and diagnostic pathology for medical AI applications.",
            requirements=["MD with Pathology residency or PhD in Pathology", "Experience with diagnostic pathology"],
            responsibilities=["Train AI on pathology diagnosis", "Annotate histopathology images", "Evaluate diagnostic AI outputs", "Design pathology benchmarks"],
            tags=["Healthcare", "Medical", "Pathology", "AI Trainer", "RLHF", "Domain Expert", "Invisible Technologies", "Meridial"]
        ),
        JobPosting(
            title="Medical AI Trainer - Clinical Diagnostics",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/invisibletech#diagnostics",
            source="Meridial (Invisible Technologies)",
            compensation="$40-$75/hr",
            job_type="Freelance/Contract",
            description="Clinical diagnostics expert training AI on differential diagnosis, lab interpretation, and diagnostic reasoning chains.",
            requirements=["MD or DO degree", "Clinical diagnostic experience", "Board certification preferred"],
            responsibilities=["Train AI on diagnostic reasoning", "Evaluate differential diagnosis outputs", "Annotate clinical cases", "Review lab interpretation"],
            tags=["Healthcare", "Medical", "Diagnostics", "AI Trainer", "RLHF", "Domain Expert", "Invisible Technologies", "Meridial"]
        ),
        JobPosting(
            title="Medical AI Trainer - Medical Ethics",
            company="Invisible Technologies + Meridial",
            location="Remote",
            url="https://job-boards.eu.greenhouse.io/invisibletech#ethics",
            source="Meridial (Invisible Technologies)",
            compensation="$35-$60/hr",
            job_type="Freelance/Contract",
            description="Medical ethics specialist evaluating AI outputs for ethical medical reasoning, patient autonomy, informed consent, and end-of-life care considerations.",
            requirements=["Advanced degree in Bioethics or Medical Ethics", "Clinical ethics committee experience preferred"],
            responsibilities=["Evaluate AI ethical reasoning", "Annotate ethics edge cases", "Design medical ethics benchmarks", "Train AI on ethical guidelines"],
            tags=["Healthcare", "Medical", "Ethics", "Bioethics", "AI Trainer", "RLHF", "Domain Expert", "Invisible Technologies", "Meridial"]
        ),
        JobPosting(
            title="Clinical Data Annotation Lead",
            company="Innodata",
            location="Remote",
            url="https://www.innodata.com/careers#clinical-annotation",
            source="Innodata Careers",
            compensation="$45-$65/hr",
            job_type="Contract",
            description="Lead clinical data annotation projects for healthcare AI, including radiology, pathology, and EHR data annotation. Manage annotation quality and team of medical annotators.",
            requirements=["Clinical background (RN, MD, or allied health)", "2+ years annotation/data experience", "Knowledge of medical terminology and coding (ICD-10, CPT)"],
            responsibilities=["Lead clinical annotation projects", "Design annotation guidelines", "QA medical annotations", "Train junior annotators"],
            tags=["Healthcare", "Medical", "Clinical", "Data Annotation", "Radiology", "Pathology", "Lead", "Innodata"]
        ),
        JobPosting(
            title="Senior Oncology Data Abstractor",
            company="Centific",
            location="Remote",
            url="https://www.centific.com/careers#oncology",
            source="Centific Careers",
            compensation="$30-$51/hr",
            job_type="Contract",
            description="Abstract and annotate oncology clinical data for AI model training. Requires expertise in cancer staging, treatment protocols, and oncology terminology.",
            requirements=["RN with oncology experience or Certified Tumor Registrar (CTR)", "Knowledge of cancer staging systems (TNM, AJCC)", "EHR experience"],
            responsibilities=["Abstract oncology records", "Annotate cancer treatment data", "QA oncology annotations", "Apply staging classifications"],
            tags=["Healthcare", "Medical", "Oncology", "Data Abstraction", "AI Training", "Domain Expert", "Centific"]
        ),
        JobPosting(
            title="Critical Care RN - AI Annotation Specialist",
            company="Centific",
            location="Remote",
            url="https://www.centific.com/careers#critical-care",
            source="Centific Careers",
            compensation="$35-$55/hr",
            job_type="Contract",
            description="Critical care nurse annotating ICU clinical data for AI applications including ventilator management, sepsis detection, and critical care decision support.",
            requirements=["Active RN license", "3+ years ICU/critical care experience", "CCRN certification preferred"],
            responsibilities=["Annotate ICU clinical data", "Evaluate AI clinical outputs", "Design critical care benchmarks", "Provide nursing expertise"],
            tags=["Healthcare", "Medical", "Nursing", "Critical Care", "ICU", "AI Annotation", "Domain Expert", "Centific"]
        ),

        # ============================================================
        # Environment Building & Experience Curation Companies
        # ============================================================

        # Bespoke Labs — Data curation platform ($7.25M Series A)
        JobPosting(
            title="Human Data for RL",
            company="Bespoke Labs",
            location="Remote",
            url="https://jobs.ashbyhq.com/bespokelabs",
            source="Bespoke Labs Careers (Ashby)",
            compensation="$25-$40/hr",
            job_type="Contract",
            description="Design original, challenging problems across technical domains (data science, systems, networking, software development, DevOps scenarios) for reinforcement learning training data. Bespoke Labs builds data curation tools for AI model development including OpenThoughts dataset.",
            requirements=["Technical expertise in data science, DevOps, networking, or software development", "Ability to design novel problems", "Strong analytical skills"],
            responsibilities=["Design challenging technical problems", "Create RL training scenarios", "Evaluate problem quality", "Contribute to training data pipelines"],
            tags=["RL", "Data Science", "DevOps", "Software Engineering", "AI Training", "Environment Building", "Bespoke Labs"]
        ),

        # Mercor + Sepal AI — Expert network (Mercor $10B, acquired Sepal Feb 2026)
        JobPosting(
            title="Expert Network - Domain Specialist (PhD/Professional)",
            company="Mercor + Sepal AI",
            location="Remote",
            url="https://www.sepalai.com/experts",
            source="Sepal AI Expert Network (Mercor)",
            compensation="~$30+/hr (varies by domain)",
            job_type="Freelance/Contract",
            description="Join Sepal AI's expert network (20k+ experts) for paid domain-specific annotation and evaluation work shaping frontier LLM capabilities. Sepal AI was acquired by Mercor ($10B valuation) in Feb 2026. Short remote projects, performance bonuses available.",
            requirements=["PhD, professional degree, or deep domain expertise", "Academic or industry experience in a specialized field"],
            responsibilities=["Answer domain-specific questions", "Evaluate AI outputs in your specialty", "Provide expert feedback on model capabilities", "Contribute to training data and benchmarks"],
            tags=["Domain Expert", "RLHF", "AI Training", "Expert Network", "Mercor", "Sepal AI", "Environment Building"]
        ),

        # Phinity Labs — Semiconductor/hardware RL environments ($5.5M seed)
        JobPosting(
            title="Senior Founding RTL Design Engineer",
            company="Phinity Labs",
            location="San Francisco, CA",
            url="https://wellfound.com/jobs/3540724-senior-founding-rtl-design-engineer",
            source="Wellfound (Phinity Labs)",
            compensation="$180K-$240K + 0.1-0.6% equity",
            job_type="Full-Time",
            description="Founding RTL design engineer building RL environments and expert data labeling for AI to master semiconductor design, verification, debugging, and P&R. Phinity works with frontier model labs to build realistic chip design training environments. Founders trained best open-source frontier models in RTL code gen at NVIDIA.",
            requirements=["Deep RTL design expertise", "Experience with semiconductor verification and debugging", "Strong systems engineering background"],
            responsibilities=["Build RL environments for chip design AI", "Create expert training data for RTL workflows", "Design verification and debugging scenarios", "Collaborate with frontier model labs"],
            tags=["Semiconductor", "RTL", "Hardware", "Engineering", "Domain Expert", "RL", "Environment Building", "Phinity Labs"],
            job_category="Corporate/In-House"
        ),

        # Fleet AI — Simulated worlds for AI agents (Sequoia, Menlo)
        JobPosting(
            title="Member of Technical Staff, Data",
            company="Fleet AI",
            location="San Francisco, CA / New York, NY",
            url="https://www.fleetai.com/careers",
            source="Fleet AI Careers",
            compensation="Competitive + equity",
            job_type="Full-Time",
            description="Own real-world and synthetic data pipelines for AI agent training. Develop realistic scenarios and simulations for agents. Fleet AI creates simulated worlds and real-world challenges to understand and shape AI agent behavior. Backed by Sequoia Capital and Menlo Ventures.",
            requirements=["Strong data engineering background", "Experience with simulation or synthetic data", "ML pipeline experience"],
            responsibilities=["Build data pipelines for agent training", "Design realistic simulation scenarios", "Develop synthetic data generation", "Collaborate on agent behavior research"],
            tags=["Data Engineering", "Simulation", "AI Training", "Environment Building", "Fleet AI"],
            job_category="Corporate/In-House"
        ),
    ]
