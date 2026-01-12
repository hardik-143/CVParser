from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import spacy
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CV Text Parser Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Request Models
class BasicInfo(BaseModel):
    email: Optional[str]
    phone: Optional[str]
    linkedin: Optional[str]
    github: Optional[str]

class Entities(BaseModel):
    people: List[str] = []
    places: List[str] = []
    organizations: List[str] = []
    dates: List[str] = []

class ExtractedTextRequest(BaseModel):
    raw_text: str
    page_texts: List[str] = []
    num_pages: int = 1
    basic_info: Optional[BasicInfo] = None
    sections: Dict[str, str] = {}
    entities: Optional[Entities] = None
    file_name: Optional[str] = None

# Response Models
class Education(BaseModel):
    degree: Optional[str]
    institution: Optional[str]
    year: Optional[str]
    field: Optional[str]

class Experience(BaseModel):
    title: Optional[str]
    company: Optional[str]
    duration: Optional[str]
    description: Optional[str]

class ParsedCV(BaseModel):
    name: Optional[str]
    candidate_title: str = ''
    candidate_forename: str = ''
    candidate_surname: str = ''
    email: Optional[str]
    phone: Optional[str]
    address: Optional[str]
    skills: List[str]
    education: List[Education]
    experience: List[Experience]
    summary: Optional[str]
    linkedin: Optional[str]
    github: Optional[str]
    raw_text: str
    confidence_scores: Dict[str, float] = {}

class IntelligentCVParser:
    def __init__(self):
        self.skills_database = {
            'languages': ['python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'golang', 'rust', 'swift', 'kotlin', 'scala'],
            'frontend': ['react', 'react.js', 'reactjs', 'angular', 'vue', 'vue.js', 'next.js', 'nextjs', 'svelte', 'electron', 'electron.js', 'redux', 'redux-saga', 'mobx', 'vuex', 'jquery'],
            'backend': ['node', 'node.js', 'nodejs', 'express', 'express.js', 'django', 'flask', 'fastapi', 'spring', 'laravel', 'asp.net', '.net', 'rails', 'nest.js', 'nestjs'],
            'mobile': ['react native', 'flutter', 'ios', 'android', 'xamarin', 'ionic', 'swift', 'kotlin'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle', 'sqlite', 'dynamodb', 'firebase', 'mariadb'],
            'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 'netlify', 'vercel', 'cloudflare'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions', 'travis ci', 'terraform', 'ansible', 'circleci'],
            'tools': ['git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'slack', 'vs code', 'visual studio', 'postman', 'swagger'],
            'styling': ['css', 'scss', 'sass', 'less', 'tailwind', 'tailwind css', 'bootstrap', 'material ui', 'mui', 'ant design', 'chakra ui', 'bulma', 'styled-components'],
            'testing': ['jest', 'mocha', 'chai', 'jasmine', 'pytest', 'junit', 'selenium', 'cypress', 'puppeteer', 'playwright'],
            'data': ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi', 'excel'],
            'ml': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'machine learning', 'deep learning', 'nlp', 'opencv', 'computer vision'],
            'healthcare': [
                'clinical assessment', 'patient care', 'nursing', 'prescribing', 'diagnosis',
                'treatment', 'physical examination', 'history taking', 'advanced history taking',
                'independent prescribing', 'minor illness', 'minor injury', 'management',
                'chronic disease', 'diabetes', 'asthma', 'copd', 'hypertension',
                'urgent care', 'triage', 'safeguarding', 'supervision', 'mentorship',
                'clinical supervision', 'multidisciplinary', 'teamwork', 'patient-centred',
                'governance', 'life support', 'ale', 'als', 'vaccination', 'screening',
                'nmc', 'health promotion', 'nurse', 'practitioner', 'physician', 'doctor'
            ],
            'other': ['rest api', 'restful', 'graphql', 'websocket', 'socket.io', 'mqtt', 'rtsp', 'microservices', 'jwt', 'oauth', 'webpack', 'vite', 'babel', 'nginx', 'apache', 'chart.js', 'd3.js', 'apexcharts', 'three.js']
        }
        
        # Section detection patterns
        self.section_keywords = {
            'summary': ['professional summary', 'about', 'overview', 'profile', 'introduction'],
            'experience': ['professional experience', 'work experience', 'experience', 'employment'],
            'education': ['education', 'academic', 'qualification', 'degree'],
            'skills': ['skills', 'technical skills', 'competencies', 'expertise', 'proficiencies', 'key clinical skills', 'core competencies', 'key skills', 'specialist skills'],
            'projects': ['projects', 'key projects', 'portfolio'],
            'certifications': ['certifications', 'certificates', 'achievements', 'awards', 'registrations', 'professional certifications'],
            'contact': ['contact', 'contact information', 'contact details']
        }
    
    def detect_sections(self, text: str, provided_sections: Dict[str, str]) -> Dict[str, str]:
        """Intelligently detect CV sections from text"""
        if provided_sections:
            return provided_sections
        
        detected = {}
        text_lower = text.lower()
        lines = text.split('\n')
        
        for section_type, keywords in self.section_keywords.items():
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                if any(kw in line_lower for kw in keywords):
                    # Only match if line is likely a header (short, no email/URL patterns)
                    # Skip if it's part of a paragraph (contains periods, commas at end, or very long)
                    if len(line) > 150 or '@' in line or 'http' in line:
                        continue
                    if line.rstrip().endswith((',', '.')):
                        continue
                    
                    # Found section header, extract content until next header
                    section_content = []
                    for j in range(i + 1, min(i + 50, len(lines))):
                        next_line = lines[j].strip()
                        # Check if it's another section header (must be short like actual headers)
                        is_header = False
                        if len(next_line) < 150 and '@' not in next_line and 'http' not in next_line:
                            for other_keywords in self.section_keywords.values():
                                if any(kw in next_line.lower() for kw in other_keywords):
                                    is_header = True
                                    break
                        if is_header:
                            break
                        section_content.append(next_line)
                    
                    detected[section_type] = '\n'.join(section_content)
                    break
        
        return detected
    
    def parse_name_components(self, full_name: str) -> tuple[str, str, str]:
        """Parse full name into title, forename, and surname components
        Returns: (title, forename, surname) - empty strings if not found"""
        if not full_name:
            return ('', '', '')
        
        # Common titles
        titles = ['Dr.', 'Dr', 'Prof.', 'Prof', 'Mr.', 'Mr', 'Ms.', 'Ms', 'Mrs.', 'Mrs', 'Sir', 'Lady', 'Rev.', 'Rev']
        
        words = full_name.split()
        title = ''
        
        # Check if first word is a title
        if words and any(words[0].lower().rstrip('.') == t.lower().rstrip('.') for t in titles):
            # Format title properly - remove dots and uppercase
            title = words[0].upper().rstrip('.')
            words = words[1:]
        
        # Extract forename and surname from remaining words
        if len(words) == 0:
            return (title, '', '')
        elif len(words) == 1:
            # Only one name left - treat as forename
            return (title, words[0], '')
        else:
            # Last word is surname, everything else is forename
            forename = ' '.join(words[:-1])
            surname = words[-1]
            return (title, forename, surname)
    
    def extract_name(self, text: str, entities: Optional[Entities]) -> tuple[Optional[str], float]:
        """Extract name with improved confidence scoring"""
        # Try entities first
        if entities and entities.people:
            for person in entities.people[:3]:
                if person and 2 <= len(person.split()) <= 4:
                    # Verify it looks like a name
                    words = person.split()
                    if all(w and w[0].isupper() for w in words):
                        return (person, 0.95)
        
        # Try first lines with various patterns
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        for i, line in enumerate(lines[:10]):
            # Skip lines that are too long or too short
            if len(line) < 3:
                continue
            
            # Skip lines that start with "Page" (common in extracted text)
            if line.lower().startswith('page '):
                continue
            
            # Pattern 1: All uppercase name
            if line.isupper() and 5 < len(line) < 50:
                words = line.split()
                if 2 <= len(words) <= 4 and all(w.isalpha() or w in ['-', "'"] for w in words):
                    return (line, 0.88)
            
            # Pattern 2: Title case name with possible titles/credentials (e.g., "Dr. Sarah Williams, MSc ACP, ANP")
            # Keep the original line but clean up credentials for processing
            original_line = line
            clean_line = line
            
            # Remove credential abbreviations after comma (MSc, ACP, ANP, RN, etc. but not names)
            # Only remove if they're typical credential abbreviations (known ones and exactly 3 letters)
            clean_line = re.sub(r',\s*(?:MSc|BSc|MA|BA|PhD|MBA|RN|ANP|ACP)(?:\s|,|$)', ',', clean_line)
            clean_line = re.sub(r',\s+', ', ', clean_line)  # Clean up comma spacing
            # Remove everything after comma (which now only has leftover credentials)
            if ',' in clean_line:
                clean_line = clean_line.split(',')[0]
            clean_line = clean_line.strip()
            
            if clean_line:
                words = clean_line.split()
                if 2 <= len(words) <= 5:  # Allow for title + first + last name
                    if all(w and w[0].isupper() for w in words if w.isalpha()):
                        # Use SpaCy to verify - but check the cleaned line
                        doc = nlp(clean_line)
                        for ent in doc.ents:
                            if ent.label_ == "PERSON" and len(ent.text.split()) <= 5:
                                # Return the original line (with title) if SpaCy found a person
                                return (clean_line, 0.90)
                        # If SpaCy didn't find it but looks like a name, return the cleaned line
                        if len(words) >= 2:
                            return (clean_line, 0.85)
        
        return (None, 0.0)
    
    
    def validate_and_clean_email(self, email: Optional[str]) -> tuple[Optional[str], float]:
        """Validate and clean email"""
        if not email:
            return (None, 0.0)
        
        email = email.lower().strip()
        
        # Validate format
        if '@' not in email or '.' not in email.split('@')[1]:
            return (None, 0.0)
        
        # Check for common corruptions
        if len(email) < 6 or len(email) > 100:
            return (None, 0.3)
        
        # Calculate confidence based on domain
        common_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com', 'protonmail.com']
        domain = email.split('@')[1] if '@' in email else ''
        confidence = 0.95 if domain in common_domains else 0.85
        
        return (email, confidence)
    
    def extract_email_from_text(self, text: str) -> tuple[Optional[str], float]:
        """Extract email from CV text if not provided in basic info"""
        # Email regex pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        matches = re.findall(email_pattern, text)
        
        if matches:
            # Take the first email and validate it
            email = matches[0].lower().strip()
            # Exclude common placeholder emails
            if 'example' not in email and 'test' not in email and 'sample' not in email:
                _, confidence = self.validate_and_clean_email(email)
                if email and confidence > 0:
                    return (email, confidence)
        
        return (None, 0.0)
    
    def extract_phone(self, text: str, provided_phone: Optional[str]) -> tuple[Optional[str], float]:
        """Extract phone number intelligently with flexible spacing (supports UK, US, India formats)"""
        if provided_phone:
            # Validate provided phone
            phone_clean = re.sub(r'[^\d+\-\(\)\s]', '', provided_phone).strip()
            # Clean up multiple spaces
            phone_clean = re.sub(r'\s+', ' ', phone_clean)
            if len(re.sub(r'\D', '', phone_clean)) >= 7:
                return (phone_clean, 0.95)
        
        # Search for phone in text with flexible patterns (UK, US, India)
        patterns = [
            # UK London landline: 020 XXXX XXXX or +44 20 XXXX XXXX
            r'(?:\+44\s*20|020)[\s\-]?\d{4}[\s\-]?\d{4}',
            
            # UK regional landline: 0121 XXXX or 0131 XXXX (Birmingham, Edinburgh etc)
            r'0(?:121|131|114|116|151|161|191)[\s\-]?\d{3,4}[\s\-]?\d{3,4}',
            
            # UK mobile: 07XXX XXXXXX or +44 7XXX XXXXXX
            r'(?:\+447|07)\d{3}[\s\-]?\d{3}[\s\-]?\d{3}',
            
            # UK format with flexible spacing: +44 (0) 20 or similar
            r'\+44\s*\(?0?\)?[\s\-]?\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4}',
            
            # International format with flexible spacing: +XX XXXXXXXX or +XX X XXXX X XXXX
            r'\+\d{1,3}(?:\s+|\s*\-\s*|\s*\(\s*)*[\d\s\-\(\)]{7,20}',
            
            # India format with flexible spacing: +91 XXXXXXXXXX or +91 XXXX XXX XXX
            r'\+91[\s\-]*\d[\d\s\-]*\d[\d\s\-]*\d[\d\s\-]*\d[\d\s\-]*\d[\d\s\-]*\d[\d\s\-]*\d[\d\s\-]*\d[\d\s\-]*\d',
            
            # US format with parentheses: (XXX) XXX-XXXX
            r'\(\d{3}\)[\s\-]?\d{3}[\s\-]?\d{4}',
            
            # US format: XXX-XXX-XXXX or XXX XXX XXXX
            r'\d{3}[\s\-]?\d{3}[\s\-]?\d{4}',
            
            # Generic format with spaces: +91 88848 34848 or similar
            r'\+\d{1,3}(?:\s+\d{2,5}){2,4}',
            
            # 10+ digit numbers with flexible spacing
            r'\d[\s\-\(\)]*\d[\s\-\(\)]*\d[\s\-\(\)]*\d[\s\-\(\)]*\d[\s\-\(\)]*\d[\s\-\(\)]*\d[\s\-\(\)]*\d[\s\-\(\)]*\d[\s\-\(\)]*\d(?:\s*\d+)?',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                candidate = match.group(0).strip()
                # Validate: must have at least 7 digits
                digit_count = len(re.sub(r'\D', '', candidate))
                if digit_count >= 7:
                    # Clean up excessive spacing
                    cleaned = re.sub(r'\s+', ' ', candidate)
                    return (cleaned, 0.85)
        
        return (None, 0.0)
    
    
    def extract_skills(self, text: str, sections: Dict[str, str]) -> tuple[List[str], float]:
        """Extract skills with advanced filtering and deduplication"""
        # Prioritize skills section if available, then try full text
        search_text = sections.get('skills', text)
        text_lower = search_text.lower()
        
        found_skills = {}  # {skill: (count, category)}
        
        for category, skills_list in self.skills_database.items():
            for skill in skills_list:
                # Use word boundary matching
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                matches = re.findall(pattern, text_lower)
                
                if matches:
                    # Format skill with proper capitalization
                    if skill.lower() in ['css', 'html', 'sql', 'aws', 'gcp', 'api', 'ui', 'ux', 'jwt', 'rest', 'mqtt', 'rtsp']:
                        formatted_skill = skill.upper()
                    elif '.' in skill:
                        formatted_skill = skill
                    else:
                        formatted_skill = skill.title()
                    
                    # Avoid duplicates with different capitalizations
                    found_skills[formatted_skill] = (len(matches), category)
        
        # If no skills found in sections, try searching whole text with different patterns
        if not found_skills and sections:
            text_lower = text.lower()
            for category, skills_list in self.skills_database.items():
                for skill in skills_list:
                    pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        if skill.lower() in ['css', 'html', 'sql', 'aws', 'gcp', 'api', 'ui', 'ux', 'jwt', 'rest', 'mqtt', 'rtsp']:
                            formatted_skill = skill.upper()
                        elif '.' in skill:
                            formatted_skill = skill
                        else:
                            formatted_skill = skill.title()
                        found_skills[formatted_skill] = (len(matches), category)
        
        # Calculate confidence based on skills section availability and density
        skill_density = len(found_skills) / max(len(search_text.split()), 1)
        if 'skills' in sections:
            confidence = min(0.99, 0.95 + skill_density * 0.01)
        else:
            confidence = min(0.99, 0.75 + skill_density * 0.1)
        
        # Sort by frequency and remove low-confidence duplicates
        sorted_skills = sorted(found_skills.items(), key=lambda x: x[1][0], reverse=True)
        unique_skills = list(dict(sorted_skills).keys())
        
        return (unique_skills, confidence)
    
    
    def parse_education(self, sections: Dict[str, str], text: str) -> tuple[List[Education], float]:
        """Parse education with adaptive patterns for different formats (supports UK, US, India standards)"""
        education_list = []
        search_text = sections.get('education', text)
        
        # Enhanced degree patterns (UK, US, India formats)
        patterns = [
            # UK A-Levels - must come FIRST (colon or space separated like "A-Levels: Maths, Physics")
            (r'A[- ]?Levels?\s*[:\s-]+\s*([A-Za-z,\s&0-9]+?)(?=\s*(?:St\.|School|College|20\d{2}|Grade|-|\n|$))', 'A-Level'),
            
            # MSc explicitly (Master of Science) - require at least 2 characters to avoid matching partial words
            (r'\bMSc\.?\s+(?:in\s+)?([A-Za-z]+(?:\s+[A-Za-z&,]+)*)(?=\s*(?:from|at|University|College|—|–|-|\d{4}|$))', 'Master'),
            
            # UK/US Bachelor's degrees: BSc, BA, BEng, BArch, B.Tech, BE (allow Hons designation)
            (r'\bB(?:Sc|Eng|Arch|Tech|E|S)\.?\s*(?:\(Hons\))?\s*(?:in|of)?\s*([A-Za-z\s&,]+?)(?=\s*(?:from|at|University|College|—|–|-|\d{4}|$))', 'Bachelor'),
            
            # BA specifically (Bachelor of Arts) - avoid matching MA
            (r'\bBA\.?\s+(?:\(Hons\))?\s*(?:in|of)?\s*([A-Za-z\s&,]+?)(?=\s*(?:from|at|University|College|—|–|-|\d{4}|$))', 'Bachelor'),
            
            # MA specifically (Master of Arts)
            (r'\bMA\.?\s+(?:in|of)?\s*([A-Za-z\s&,]+?)(?=\s*(?:from|at|University|College|—|–|-|\d{4}|$))', 'Master'),
            
            # MEng, MBA, other Master degrees
            (r'\bM(?:Eng|BA|Tech|E)(?:\b|\.)\s*(?:in|of)?\s*([A-Za-z\s&,]+?)(?=\s*(?:from|at|University|College|—|–|-|\d{4}|$))', 'Master'),
            
            # PhD/Doctorate
            (r'(?:Ph\.?D\.?|Doctorate)(?:\s+(?:in|of))?\s*([A-Za-z\s&,]+?)(?=\s*(?:from|at|University|College|—|–|-|$))', 'PhD'),
            
            # UK GCSEs
            (r'GCSE?s?\s*[:–—-]\s*([A-Za-z,\s&0-9]+?)(?=\s*(?:Grade|20\d{2}|$|—|–))', 'GCSE'),
            
            # UK HND/HNC (Higher National Diploma/Certificate)
            (r'HN[DC](?:\s+(?:in|of))?\s+([A-Za-z\s&,]+?)(?=\s*(?:from|at|University|College|—|–|-|$))', 'HND/HNC'),
            
            # Generic Bachelor pattern (fallback)
            (r'Bachelor(?:\s+of)?(?:\s+(?:Science|Arts|Engineering|Technology|Architecture))?(?:\s+(?:in|of))?\s+([A-Za-z\s&,]+?)(?=\s*(?:from|at|—|–|-|$))', 'Bachelor'),
            
            # Generic Master pattern (fallback)
            (r'Master(?:\s+of)?(?:\s+(?:Science|Arts|Engineering|Business|Technology))?(?:\s+(?:in|of))?\s+([A-Za-z\s&,]+?)(?=\s*(?:from|at|—|–|-|$))', 'Master'),
            
            # PUC/Intermediate education
            (r'(?:PUC|Pre-University|Intermediate)\s*(?:–|-|—)?\s*([A-Za-z\s]+?)(?=\s*(?:\d{4}|$|\n))', 'PUC'),
            
            # High School/Class 10
            (r'(?:CBSE|Class\s+10|High\s+School|Secondary|SSLC)\s*(?:–|-|—)?\s*([A-Za-z\s]+?)(?=\s*(?:\d{4}|%|$|\n))', 'High School'),
        ]
        
        seen_educations = set()  # Track duplicates
        
        for pattern, degree_type in patterns:
            for match in re.finditer(pattern, search_text, re.IGNORECASE):
                field = match.group(1).strip() if match.lastindex >= 1 else None
                
                if not field:
                    continue
                
                # Clean field
                field = re.sub(r'\s+', ' ', field)
                field = re.split(r'[\n(]', field)[0].strip()
                field = field.rstrip('.,;')
                
                # Skip if field is too short or too long
                if len(field) < 3 or len(field) > 100:
                    continue
                
                # Get context for year and institution - smaller context window
                context_start = max(0, match.start() - 50)
                context_end = min(len(search_text), match.end() + 150)
                context = search_text[context_start:context_end]
                
                # Extract year with various formats
                year = None
                year_patterns = [
                    r'(20\d{2}|19\d{2})\s*(?:–|-|to|—)\s*(?:(20\d{2}|19\d{2})|Present|present)',
                    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2})',
                    r'(20\d{2}|19\d{2})',
                ]
                
                for year_pattern in year_patterns:
                    year_match = re.search(year_pattern, context, re.IGNORECASE)
                    if year_match:
                        year = year_match.group(0)
                        break
                
                # Extract institution
                institution = None
                context_lines = context.split('\n')
                institution_keywords = ['university', 'college', 'institute', 'school', 'academy', 'vidyalaya', 'high school', 'government', 'private']
                
                                # Find the line containing the degree match
                match_line_index = None
                for i, line in enumerate(context_lines):
                    if match.group(0) in line:
                        match_line_index = i
                        break
                
                if match_line_index is not None:
                    # Look for institution in the next few lines (max 3 lines after the degree)
                    for i in range(match_line_index + 1, min(match_line_index + 4, len(context_lines))):
                        line = context_lines[i].strip()
                        if line and len(line) > 8 and any(kw in line.lower() for kw in institution_keywords):
                            # Stop if we hit another degree pattern (avoid cross-contamination)
                            # Only stop if the line actually starts with a degree pattern, not just contains keywords
                            line_starts_with_degree = False
                            for p, _ in patterns:
                                if re.match(p, line, re.IGNORECASE):
                                    line_starts_with_degree = True
                                    break
                            if line_starts_with_degree:
                                break
                            # Check if this line contains year information - if so, it's likely not just institution
                            if not re.search(r'\d{4}', line):
                                institution = line
                                break
                            else:
                                # Line has year, extract institution part before year
                                parts = re.split(r'\d{4}', line)
                                if parts[0].strip() and any(kw in parts[0].lower() for kw in institution_keywords):
                                    institution = parts[0].strip()
                                    break
                
                # Clean institution name if found
                if institution:
                    institution = re.sub(r'\s+', ' ', institution).strip()
                    # Clean institution name - remove dates, grades, etc.
                    institution = re.split(r'[–—(-]', institution)[0].strip()
                    institution = re.split(r'\d{4}', institution)[0].strip()  # Remove years
                    # Remove common suffixes
                    institution = re.sub(r'\s+(?:CGPA|GPA|CPI|Grade).*', '', institution, flags=re.IGNORECASE).strip()
                    # Remove trailing punctuation
                    institution = institution.rstrip('.,;:-')
                    if len(institution) <= 5 or len(institution) > 200:
                        institution = None
                
                # Create unique key to avoid duplicates
                edu_key = (degree_type, field, institution, year)
                if edu_key not in seen_educations:
                    education_list.append(Education(
                        degree=degree_type,
                        field=field,
                        year=year,
                        institution=institution
                    ))
                    seen_educations.add(edu_key)
        
        confidence = 0.95 if 'education' in sections and education_list else 0.75
        return (education_list, confidence)
    
    def convert_to_array_format(self, parsed_cv: ParsedCV) -> List[Dict[str, Any]]:
        """Convert parsed CV data to array format"""
        data_array = []
        
        # Basic information
        if parsed_cv.name:
            data_array.append({
                "type": "personal_info",
                "field": "name",
                "value": parsed_cv.name,
                "confidence": parsed_cv.confidence_scores.get('name', 0.0)
            })
        
        if parsed_cv.candidate_title:
            data_array.append({
                "type": "personal_info",
                "field": "title",
                "value": parsed_cv.candidate_title,
                "confidence": parsed_cv.confidence_scores.get('name', 0.0)
            })
        
        if parsed_cv.candidate_forename:
            data_array.append({
                "type": "personal_info",
                "field": "forename",
                "value": parsed_cv.candidate_forename,
                "confidence": parsed_cv.confidence_scores.get('name', 0.0)
            })
        
        if parsed_cv.candidate_surname:
            data_array.append({
                "type": "personal_info",
                "field": "surname",
                "value": parsed_cv.candidate_surname,
                "confidence": parsed_cv.confidence_scores.get('name', 0.0)
            })
        
        if parsed_cv.email:
            data_array.append({
                "type": "contact_info",
                "field": "email",
                "value": parsed_cv.email,
                "confidence": parsed_cv.confidence_scores.get('email', 0.0)
            })
        
        if parsed_cv.phone:
            data_array.append({
                "type": "contact_info",
                "field": "phone",
                "value": parsed_cv.phone,
                "confidence": parsed_cv.confidence_scores.get('phone', 0.0)
            })
        
        if parsed_cv.linkedin:
            data_array.append({
                "type": "contact_info",
                "field": "linkedin",
                "value": parsed_cv.linkedin,
                "confidence": parsed_cv.confidence_scores.get('linkedin', 0.0)
            })
        
        if parsed_cv.github:
            data_array.append({
                "type": "contact_info",
                "field": "github",
                "value": parsed_cv.github,
                "confidence": parsed_cv.confidence_scores.get('github', 0.0)
            })
        
        # Skills
        for skill in parsed_cv.skills:
            data_array.append({
                "type": "skills",
                "field": "skill",
                "value": skill,
                "confidence": parsed_cv.confidence_scores.get('skills', 0.0)
            })
        
        # Education
        for edu in parsed_cv.education:
            data_array.append({
                "type": "education",
                "field": "degree",
                "value": edu.degree,
                "details": {
                    "field": edu.field,
                    "institution": edu.institution,
                    "year": edu.year
                },
                "confidence": parsed_cv.confidence_scores.get('education', 0.0)
            })
        
        # Experience
        for exp in parsed_cv.experience:
            data_array.append({
                "type": "experience",
                "field": "job",
                "value": exp.title,
                "details": {
                    "company": exp.company,
                    "duration": exp.duration,
                    "description": exp.description
                },
                "confidence": parsed_cv.confidence_scores.get('experience', 0.0)
            })
        
        # Summary
        if parsed_cv.summary:
            data_array.append({
                "type": "summary",
                "field": "professional_summary",
                "value": parsed_cv.summary,
                "confidence": parsed_cv.confidence_scores.get('summary', 0.0)
            })
        
        return data_array
    
    
    def parse_experience(self, sections: Dict[str, str], text: str) -> tuple[List[Experience], float]:
        """Parse work experience with intelligent formatting detection"""
        experience_list = []
        search_text = sections.get('experience', text)
        
        # Lines to process
        lines = [l.strip() for l in search_text.split('\n') if l.strip()]
        
        job_keywords = [
            'developer', 'engineer', 'designer', 'manager', 'analyst', 'consultant',
            'architect', 'specialist', 'lead', 'senior', 'junior', 'intern', 'programmer',
            'administrator', 'coordinator', 'director', 'officer', 'associate', 'assistant',
            'devops', 'qa', 'tester', 'technician', 'support',
            # Healthcare roles
            'nurse', 'practitioner', 'doctor', 'physician', 'surgeon', 'therapist',
            'dentist', 'pharmacist', 'clinician', 'consultant', 'specialist'
        ]
        
        company_keywords = ['pvt', 'ltd', 'inc', 'corp', 'corporation', 'company', 'studio', 'agency', 'solutions', 'technology', 'software', 'services', 'group', 'enterprises', 'tekno']
        
        months_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*/\d{4}|\d{4}'
        
        seen_experiences = set()
        i = 0
        
        while i < len(lines):
            line = lines[i]
            has_job_keyword = any(kw in line.lower() for kw in job_keywords)
            has_date = bool(re.search(months_pattern, line, re.IGNORECASE))
            has_company_keyword = any(kw in line.lower() for kw in company_keywords)
            
            # Pattern 0: Job Title – Company Format (with em-dash)
            # Example: "Advanced Nurse Practitioner – NHS GP Practice, Manchester"
            # Followed by date on next line: "March 2018 – Present"
            if '–' in line and has_job_keyword and not has_date:
                parts = line.split('–')
                if len(parts) >= 2:
                    title = parts[0].strip()
                    company_part = parts[1].strip()
                    
                    # Check if next line has date
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        duration_match = re.search(
                            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\s*[-–—]*\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|Present|present)',
                            next_line, re.IGNORECASE
                        )
                        
                        if duration_match:
                            duration = duration_match.group(0)
                            company = company_part
                            
                            # Collect description (bullet points from following lines)
                            description_parts = []
                            j = i + 2
                            while j < len(lines) and len(description_parts) < 4:
                                desc_line = lines[j]
                                
                                # Stop if we hit another job entry (has job keyword and em-dash)
                                if '–' in desc_line and any(kw in desc_line.lower() for kw in job_keywords):
                                    break
                                
                                # Include bullet points and descriptions
                                if desc_line and len(desc_line) > 3 and (desc_line.startswith('•') or desc_line.startswith('-')):
                                    description_parts.append(desc_line.lstrip('•-').strip())
                                elif desc_line and len(desc_line) > 10 and not any(kw in desc_line.lower() for kw in job_keywords):
                                    description_parts.append(desc_line)
                                
                                j += 1
                            
                            description = ' '.join(description_parts)[:250] if description_parts else None
                            
                            exp_key = (title, company, duration)
                            if exp_key not in seen_experiences:
                                experience_list.append(Experience(
                                    title=title,
                                    company=company,
                                    duration=duration,
                                    description=description if description and len(description) > 5 else None
                                ))
                                seen_experiences.add(exp_key)
                            
                            i = j
                            continue
            
            # Pattern 1: Company with date on first line, Title on next
            # Example: "Talbotiq Pvt Ltd Apr 2024 – Oct 2025"
            # "Full Stack Developer" (next line)
            if (has_company_keyword and has_date) and i + 1 < len(lines):
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                
                # Check if next line is a job title
                if any(kw in next_line.lower() for kw in job_keywords):
                    company_line = line
                    title = next_line
                    duration = None
                    
                    # Extract duration from company line
                    duration_match = re.search(
                        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\s*[-–to]*\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|Present|present)',
                        company_line, re.IGNORECASE
                    )
                    
                    if duration_match:
                        duration = duration_match.group(0)
                        company = company_line.replace(duration, '').strip()
                    else:
                        company = company_line
                    
                    # Collect description
                    description_parts = []
                    j = i + 2
                    while j < len(lines) and len(description_parts) < 4:
                        desc_line = lines[j]
                        
                        # Stop if we hit another company/job entry
                        if (any(kw in desc_line.lower() for kw in company_keywords) and re.search(months_pattern, desc_line)) or \
                           (any(kw in desc_line.lower() for kw in job_keywords) and not desc_line.startswith('–')):
                            break
                        
                        # Include bullet points and descriptions
                        if desc_line and len(desc_line) > 3:
                            description_parts.append(desc_line)
                        
                        j += 1
                    
                    description = ' '.join(description_parts)[:250] if description_parts else None
                    
                    exp_key = (title, company, duration)
                    if exp_key not in seen_experiences:
                        experience_list.append(Experience(
                            title=title,
                            company=company,
                            duration=duration,
                            description=description if description and len(description) > 5 else None
                        ))
                        seen_experiences.add(exp_key)
                    
                    i = j
                    continue
            
            # Pattern 2: Title on first line with date/company
            # Example: "Software Development Engineer Sep 2023 - Present"
            # "Rapidise Technology" (next line)
            if has_job_keyword and (has_date or '|' in line):
                title_line = line
                company = None
                duration = None
                
                # Extract duration from title line
                duration_match = re.search(
                    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\s*[-–to]*\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|Present|present)',
                    title_line, re.IGNORECASE
                )
                
                if not duration_match:
                    duration_match = re.search(r'(\d{4})\s*[-–to]*\s*(\d{4}|Present|present)', title_line, re.IGNORECASE)
                
                title = title_line
                if duration_match:
                    duration = duration_match.group(0)
                    title = title_line.replace(duration, '').strip()
                
                # Check for pipe-separated format
                if '|' in title:
                    parts = [p.strip() for p in title.split('|')]
                    if len(parts) >= 2:
                        title = parts[0]
                        company = parts[1]
                
                # Look for company in next line if not found
                if not company and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not any(kw in next_line.lower() for kw in job_keywords) and len(next_line) < 100:
                        company = next_line
                        i += 1
                
                # Collect description
                description_parts = []
                j = i + 1
                while j < len(lines) and len(description_parts) < 4:
                    desc_line = lines[j]
                    
                    # Stop if we hit another job entry
                    if any(kw in desc_line.lower() for kw in job_keywords) and not desc_line.startswith('–'):
                        break
                    
                    # Include descriptions
                    if desc_line and len(desc_line) > 3:
                        description_parts.append(desc_line)
                    
                    j += 1
                
                description = ' '.join(description_parts)[:250] if description_parts else None
                
                exp_key = (title, company, duration)
                if exp_key not in seen_experiences:
                    experience_list.append(Experience(
                        title=title,
                        company=company,
                        duration=duration,
                        description=description if description and len(description) > 5 else None
                    ))
                    seen_experiences.add(exp_key)
                
                i = j
                continue
            
            i += 1
        
        confidence = 0.95 if 'experience' in sections and experience_list else 0.75
        return (experience_list[:10], confidence)
    
    
    def extract_summary(self, sections: Dict[str, str]) -> tuple[Optional[str], float]:
        """Extract professional summary"""
        if 'summary' in sections:
            summary = sections['summary'].strip()
            if len(summary) > 50:
                return (summary[:600], 0.95)
        
        return (None, 0.0)
    
    def parse_cv(self, request: ExtractedTextRequest) -> ParsedCV:
        """Main parsing function with intelligent section detection"""
        text = request.raw_text
        basic_info = request.basic_info
        entities = request.entities
        
        confidence_scores = {}
        
        # Intelligently detect and merge sections
        detected_sections = self.detect_sections(text, request.sections)
        sections = {**detected_sections, **request.sections} if request.sections else detected_sections
        
        # Extract name
        name, name_conf = self.extract_name(text, entities)
        confidence_scores['name'] = name_conf
        
        # Parse name into components (title, forename, surname)
        candidate_title, candidate_forename, candidate_surname = self.parse_name_components(name) if name else ('', '', '')
        
        # Validate and extract email
        if basic_info and basic_info.email:
            email, email_conf = self.validate_and_clean_email(basic_info.email)
        else:
            # Try to extract email from CV text if not provided
            email, email_conf = self.extract_email_from_text(text)
        confidence_scores['email'] = email_conf
        
        # Extract phone
        phone, phone_conf = self.extract_phone(text, basic_info.phone if basic_info else None)
        confidence_scores['phone'] = phone_conf
        
        # LinkedIn
        linkedin = basic_info.linkedin if basic_info else None
        if linkedin and 'linkedin.com' not in linkedin.lower():
            linkedin = f"linkedin.com/in/{linkedin}"
        confidence_scores['linkedin'] = 0.95 if linkedin else 0.0
        
        # GitHub  
        github = basic_info.github if basic_info else None
        confidence_scores['github'] = 0.95 if github else 0.0
        
        # Skills
        skills, skills_conf = self.extract_skills(text, sections)
        confidence_scores['skills'] = skills_conf
        
        # Education
        education, edu_conf = self.parse_education(sections, text)
        confidence_scores['education'] = edu_conf
        
        # Experience
        experience, exp_conf = self.parse_experience(sections, text)
        confidence_scores['experience'] = exp_conf
        
        # Summary
        summary, summary_conf = self.extract_summary(sections)
        confidence_scores['summary'] = summary_conf
        
        logger.info(f"Parsing complete. Name: {name}, Education: {len(education)}, Experience: {len(experience)}, Skills: {len(skills)}")
        
        return ParsedCV(
            name=name,
            candidate_title=candidate_title,
            candidate_forename=candidate_forename,
            candidate_surname=candidate_surname,
            email=email,
            phone=phone,
            address=None,
            skills=skills,
            education=education,
            experience=experience,
            summary=summary,
            linkedin=linkedin,
            github=github,
            raw_text=text[:3000],
            confidence_scores=confidence_scores
        )


parser = IntelligentCVParser()

@app.post("/parse-extracted-cv")
async def parse_extracted_cv(request: ExtractedTextRequest):
    """Parse pre-extracted CV text and return data in array format"""
    try:
        if not request.raw_text or len(request.raw_text) < 100:
            raise HTTPException(status_code=400, detail="Insufficient text provided")
        
        logger.info(f"Processing CV: {request.file_name}, {len(request.raw_text)} chars")
        result = parser.parse_cv(request)
        
        # Always return array format
        data_array = parser.convert_to_array_format(result)
        return {"data": data_array}
        
    except Exception as e:
        logger.error(f"Parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# render html file from /templates
@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Intelligent CV Text Parser",
        "version": "2.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
