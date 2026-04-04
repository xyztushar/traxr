"""
TRAXR — Demo Data Fallback
Complete hardcoded data for a fully functional demo when LLM / PDF parsing fails.
All data is internally consistent: Priya Sharma applying for SDE Intern at TechCorp.
"""

DEMO_RESUME_TEXT = """Priya Sharma
B.Tech Computer Science | Expected May 2026
GitHub: github.com/priyasharma | LinkedIn: linkedin.com/in/priyasharma
Email: priya.sharma@email.com | Phone: +91-9876543210

TECHNICAL SKILLS
Languages: Python, JavaScript, C++, SQL
Frameworks: React, Flask, Express.js, Bootstrap
Tools: Git, Docker, VS Code, Postman, Figma
Databases: MySQL, MongoDB
Other: REST APIs, Agile, Unit Testing

PROJECTS
TaskFlow – Team Productivity Dashboard (Oct 2025)
• Built a full-stack task management app using React frontend and Flask backend
• Implemented REST API endpoints for CRUD operations on tasks and users
• Used MySQL database with SQLAlchemy ORM for data persistence
• Deployed on Railway with Docker containerization
• Achieved 95% test coverage using pytest
• GitHub: github.com/priyasharma/taskflow

SmartBudget – Personal Finance Tracker (Jul 2025)
• Developed a responsive web app for expense tracking using React and Chart.js
• Built Node.js/Express backend with MongoDB for data storage
• Implemented JWT authentication and role-based access control
• Integrated Razorpay payment gateway for premium features

CodeQuiz – Interactive Coding Practice Platform (Mar 2025)
• Created a platform for practicing coding problems with real-time code execution
• Used Python Flask with WebSocket for live code evaluation
• Built 50+ coding challenges with automated test cases
• Served 200+ active users during college coding week

EXPERIENCE
Software Engineering Intern | TechStartup Inc. (Jun 2025 – Aug 2025)
• Worked on backend API development using Python and Flask
• Reduced API response time by 40% through query optimization
• Wrote unit tests and integration tests increasing coverage from 60% to 85%
• Participated in daily standups and sprint planning (Agile/Scrum)

EDUCATION
B.Tech Computer Science and Engineering
State Institute of Technology | CGPA: 8.7/10 | 2022–2026

CERTIFICATIONS
• AWS Cloud Practitioner (2025)
• Python for Data Science – Coursera (2024)

ACHIEVEMENTS
• Winner, Smart India Hackathon 2025 – Regional Round
• 5-star rating on HackerRank (Problem Solving)
• Published technical blog with 10K+ views on Medium
"""

DEMO_JD_TEXT = """Software Development Engineer Intern – Summer 2026
TechCorp India | Bengaluru, India

About the Role:
We are looking for a passionate SDE Intern to join our Platform Engineering team.
You will work on building and scaling microservices that power our e-commerce platform
serving 10M+ users. This is a high-impact role where you will ship production code
from week one.

Required Skills:
• Strong proficiency in Python and/or Java
• Experience with React or similar frontend framework
• Knowledge of RESTful API design and implementation
• Familiarity with SQL databases (PostgreSQL preferred)
• Understanding of Git version control and code review practices
• Basic knowledge of Docker and containerization
• Exposure to cloud platforms (AWS preferred)
• Understanding of CI/CD pipelines
• Familiarity with Agile/Scrum methodology

Preferred Skills:
• Experience with Node.js or Go
• Knowledge of Kubernetes and container orchestration
• Experience with message queues (RabbitMQ/Kafka)
• Familiarity with monitoring tools (Prometheus, Grafana)
• Understanding of system design principles
• Experience with TypeScript

Day-to-Day:
• Write clean, tested, production-ready code
• Participate in code reviews and design discussions
• Collaborate with cross-functional teams
• Debug and resolve production issues
• Contribute to architectural decisions for new features
"""

DEMO_ROLE_DNA = {
    "required_skills": [
        "python", "java", "react", "rest apis", "postgresql", "git",
        "docker", "aws", "ci/cd", "agile",
    ],
    "preferred_skills": [
        "node.js", "kubernetes", "rabbitmq", "kafka", "prometheus",
        "grafana", "system design", "typescript",
    ],
    "tools": ["git", "docker", "aws", "postgresql", "github actions", "jira", "postman"],
    "behavioral_requirements": [
        "collaboration", "communication", "problem solving",
        "code review participation", "ownership mindset",
    ],
    "role_context": (
        "Platform Engineering team building microservices for a 10M+ user "
        "e-commerce platform. High-impact internship with production code "
        "deployment from week one."
    ),
    "experience_level": "intern",
    "day_to_day_activities": [
        "Write clean, tested, production-ready code",
        "Participate in code reviews and design discussions",
        "Collaborate with cross-functional teams",
        "Debug and resolve production issues",
        "Contribute to architectural decisions",
    ],
}

DEMO_PROFILE_SIGNALS = {
    "claimed_skills": [
        "python", "javascript", "c++", "sql", "react", "flask",
        "express.js", "bootstrap", "git", "docker", "postman",
        "mysql", "mongodb", "rest apis", "agile", "unit testing",
    ],
    "supported_skills": [
        "python", "react", "flask", "rest apis", "docker", "git",
        "mysql", "mongodb", "unit testing", "agile",
    ],
    "projects": [
        {
            "name": "TaskFlow – Team Productivity Dashboard",
            "tech": ["react", "flask", "mysql", "docker", "rest apis"],
            "description": "Full-stack task management with CRUD API, Docker deployment, 95% test coverage",
            "has_metrics": True, "has_link": True,
        },
        {
            "name": "SmartBudget – Personal Finance Tracker",
            "tech": ["react", "node.js", "express.js", "mongodb"],
            "description": "Responsive expense tracker with JWT auth and payment integration",
            "has_metrics": False, "has_link": False,
        },
        {
            "name": "CodeQuiz – Interactive Coding Platform",
            "tech": ["python", "flask", "websockets"],
            "description": "Coding practice platform with live execution, 200+ users",
            "has_metrics": True, "has_link": False,
        },
    ],
    "certifications": ["AWS Cloud Practitioner", "Python for Data Science – Coursera"],
    "links": ["github.com/priyasharma", "linkedin.com/in/priyasharma", "github.com/priyasharma/taskflow"],
    "education": ["B.Tech Computer Science and Engineering"],
    "experience_months": 3,
    "has_internship": True,
}

DEMO_SCORES = {
    "dimensions": {
        "Technical Skill Match": 68,
        "Evidence Quality": 72,
        "Communication Clarity": 65,
        "Role Alignment": 70,
        "Learning Momentum": 74,
        "Interview Readiness": 62,
    },
    "overall_score": 68,
    "dimension_explanations": {
        "Technical Skill Match": {
            "score": 68,
            "why": "Matched 7 of 10 required skills. Missing Java, PostgreSQL, and CI/CD pipeline experience.",
            "evidence_found": ["python ✓", "react ✓", "rest apis ✓", "git ✓", "docker ✓", "agile ✓", "sql ✓ (MySQL)"],
            "missing": ["java", "postgresql", "ci/cd"],
            "action": "Focus on Java fundamentals and build one project with PostgreSQL.",
        },
        "Evidence Quality": {
            "score": 72,
            "why": "3 projects with descriptions. TaskFlow has strong evidence. Other projects lack deployed links.",
            "evidence_found": ["3 project descriptions", "1 GitHub link", "Metrics in 2 projects", "1 internship with impact"],
            "missing": ["Live URLs for all projects", "More outcome-driven descriptions"],
            "action": "Deploy SmartBudget and CodeQuiz publicly. Add GitHub links to all projects.",
        },
        "Communication Clarity": {
            "score": 65,
            "why": "Good use of action verbs. Some bullets lack specificity and could be more outcome-focused.",
            "evidence_found": ["Active voice in 70% of bullets", "Quantified internship impact"],
            "missing": ["Vague bullets like 'Worked on backend'", "Missing team size context"],
            "action": "Rewrite vague bullets with STAR format. Quantify every achievement.",
        },
        "Role Alignment": {
            "score": 70,
            "why": "Profile aligns with backend/fullstack work. Gap in enterprise-scale and cloud-native experience.",
            "evidence_found": ["Backend + frontend experience", "Team collaboration in internship"],
            "missing": ["No microservices experience", "Limited cloud-native exposure"],
            "action": "Add a microservices project or contribute to an open-source one.",
        },
        "Learning Momentum": {
            "score": 74,
            "why": "Active project history. AWS cert shows initiative. Hackathon winner adds signal.",
            "evidence_found": ["3 projects in 7 months", "Recent certification", "Hackathon win"],
            "missing": ["No advanced certifications", "No open-source contributions"],
            "action": "Pursue AWS Solutions Architect Associate.",
        },
        "Interview Readiness": {
            "score": 62,
            "why": "Good project depth for storytelling. Gaps in system design and behavioral scenario coverage.",
            "evidence_found": ["Internship STAR stories available", "HackerRank rating"],
            "missing": ["No system design experience", "Need stronger 'why this role' narrative"],
            "action": "Prepare 3 STAR stories. Study basic system design patterns.",
        },
    },
    "confidence": "medium-high",
}

DEMO_GAPS = [
    {
        "skill": "java", "type": "hard", "severity": "high",
        "category": "Programming Languages",
        "explanation": "Java is required but completely absent from your resume.",
        "action": "Complete a Java fundamentals course and build one Spring Boot REST API.",
    },
    {
        "skill": "postgresql", "type": "context", "severity": "high",
        "category": "Data & ML",
        "explanation": "You have SQL experience with MySQL, but the role requires PostgreSQL specifically.",
        "action": "Migrate one project from MySQL to PostgreSQL.",
    },
    {
        "skill": "kubernetes", "type": "hard", "severity": "medium",
        "category": "Cloud & DevOps",
        "explanation": "Kubernetes is preferred. Docker experience is a good foundation but K8s is missing.",
        "action": "Deploy your Docker project on a local K8s cluster (minikube).",
    },
    {
        "skill": "ci/cd", "type": "soft", "severity": "medium",
        "category": "Cloud & DevOps",
        "explanation": "CI/CD is mentioned in skills but no pipeline experience is demonstrated.",
        "action": "Add a GitHub Actions workflow to TaskFlow with automated tests + deploy.",
    },
    {
        "skill": "aws", "type": "soft", "severity": "medium",
        "category": "Cloud & DevOps",
        "explanation": "You have AWS Cloud Practitioner cert but no hands-on AWS project shown.",
        "action": "Deploy one project on AWS (EC2 or Lambda).",
    },
    {
        "skill": "system design", "type": "hard", "severity": "low",
        "category": "Tools & Practices",
        "explanation": "System design is preferred. No evidence of design thinking in projects.",
        "action": "Add architecture diagrams to project READMEs.",
    },
    {
        "skill": "typescript", "type": "hard", "severity": "low",
        "category": "Programming Languages",
        "explanation": "TypeScript is preferred. Your projects use plain JavaScript.",
        "action": "Convert one React project to TypeScript.",
    },
]

DEMO_ROADMAP = [
    {
        "week": 1, "title": "Foundation: Java + PostgreSQL",
        "days": [
            {"day": "1-2", "goal": "Java fundamentals crash course",
             "tasks": ["Complete Java basics: OOP, collections, streams", "Solve 10 LeetCode Easy in Java"],
             "resources": [{"name": "Java Tutorial - W3Schools", "url": "https://www.w3schools.com/java/"},
                           {"name": "Java Crash Course - freeCodeCamp (YouTube)", "url": "https://www.youtube.com/watch?v=GdzRzWymT4c"}],
             "proof": "Screenshot of 10 solved LC problems in Java"},
            {"day": "3-4", "goal": "Build a Spring Boot REST API",
             "tasks": ["Set up Spring Boot with Spring Initializr", "Build CRUD REST API with JUnit tests"],
             "resources": [{"name": "Spring Boot Guide", "url": "https://spring.io/guides/gs/rest-service"}],
             "proof": "Push Spring Boot project to GitHub with passing tests"},
            {"day": "5-7", "goal": "PostgreSQL migration",
             "tasks": ["Install PostgreSQL", "Migrate TaskFlow from MySQL to PostgreSQL", "Learn JSONB, CTEs, window functions"],
             "resources": [{"name": "PostgreSQL Tutorial", "url": "https://www.postgresqltutorial.com/"}],
             "proof": "TaskFlow running on PostgreSQL with updated README"},
        ],
    },
    {
        "week": 2, "title": "Cloud & DevOps: AWS + CI/CD",
        "days": [
            {"day": "8-9", "goal": "Deploy on AWS",
             "tasks": ["Deploy TaskFlow backend on AWS EC2 or Lambda", "Set up RDS PostgreSQL"],
             "resources": [{"name": "AWS Free Tier", "url": "https://aws.amazon.com/free/"}],
             "proof": "Live URL of TaskFlow on AWS"},
            {"day": "10-12", "goal": "CI/CD Pipeline",
             "tasks": ["Create GitHub Actions workflow", "Add automated testing + deployment"],
             "resources": [{"name": "GitHub Actions Docs", "url": "https://docs.github.com/en/actions"}],
             "proof": "Green CI badge on GitHub"},
            {"day": "13-14", "goal": "Kubernetes basics",
             "tasks": ["Install minikube", "Write K8s deployment YAML for TaskFlow", "Deploy and expose"],
             "resources": [{"name": "K8s Tutorial - TechWorld with Nana", "url": "https://www.youtube.com/watch?v=X48VuDVv0do"}],
             "proof": "K8s YAML in repo + running pods screenshot"},
        ],
    },
    {
        "week": 3, "title": "System Design + Evidence Polish",
        "days": [
            {"day": "15-17", "goal": "System design fundamentals",
             "tasks": ["Study load balancing, caching, CDN, replication, message queues", "Draw architecture diagram for TaskFlow"],
             "resources": [{"name": "System Design Primer", "url": "https://github.com/donnemartin/system-design-primer"}],
             "proof": "Architecture diagram in TaskFlow README"},
            {"day": "18-20", "goal": "Portfolio polish",
             "tasks": ["Deploy all projects publicly", "Add live URLs + GitHub links everywhere", "Rewrite bullets in STAR format"],
             "resources": [{"name": "STAR Method Guide", "url": "https://www.themuse.com/advice/star-interview-method"}],
             "proof": "All 3 projects live with URLs"},
            {"day": "21", "goal": "TypeScript migration",
             "tasks": ["Convert SmartBudget React code to TypeScript"],
             "resources": [{"name": "React TS Cheatsheet", "url": "https://react-typescript-cheatsheet.netlify.app/"}],
             "proof": "SmartBudget running in TypeScript"},
        ],
    },
    {
        "week": 4, "title": "Interview Prep + Final Polish",
        "days": [
            {"day": "22-24", "goal": "Behavioral prep",
             "tasks": ["Prepare 5 STAR stories", "Practice 'Tell me about yourself'", "2 mock interviews"],
             "resources": [{"name": "Pramp Mock Interviews", "url": "https://www.pramp.com/"}],
             "proof": "Written STAR stories + mock feedback"},
            {"day": "25-27", "goal": "Technical prep",
             "tasks": ["Solve 15 LC Medium problems", "SQL interview questions on HackerRank"],
             "resources": [{"name": "Blind 75 List", "url": "https://leetcode.com/discuss/general-discussion/460599/"}],
             "proof": "15 LC solutions + SQL badge"},
            {"day": "28-30", "goal": "Application ready",
             "tasks": ["Update resume with new skills/projects", "Update GitHub + LinkedIn", "Submit application"],
             "resources": [{"name": "GitHub Profile README", "url": "https://docs.github.com/en/account-and-profile"}],
             "proof": "Completed application with updated portfolio"},
        ],
    },
]

DEMO_SKILLPRINT_CHALLENGE = {
    "skill": "rest apis",
    "title": "Build a REST API Endpoint",
    "description": (
        "Write a Python function using Flask that handles GET /api/tasks.\n"
        "Return a JSON list of tasks (each with id, title, status, created_at).\n"
        "Support a `?status=` query parameter to filter.\n"
        "Return 404 with error message if no tasks match the filter."
    ),
    "starter_code": (
        "from flask import Flask, jsonify, request\n\n"
        "app = Flask(__name__)\n\n"
        "tasks = [\n"
        '    {"id": 1, "title": "Review PR", "status": "pending", "created_at": "2025-10-01"},\n'
        '    {"id": 2, "title": "Write tests", "status": "completed", "created_at": "2025-10-02"},\n'
        "]\n\n"
        "# TODO: Implement GET /api/tasks with status filtering\n"
    ),
    "rubric": [
        "Correct route definition",
        "Returns JSON with proper Content-Type",
        "Implements status query parameter filtering",
        "Handles empty filter results with 404",
        "Uses appropriate HTTP status codes",
        "Clean, readable code",
    ],
    "time_estimate": "5 minutes",
    "max_score_boost": 8,
}
