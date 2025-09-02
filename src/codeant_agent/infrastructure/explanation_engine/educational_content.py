"""
Educational Content Generator for the Explanation Engine.

This module is responsible for generating educational content to help users
understand code quality issues, antipatterns, and best practices.
"""
from typing import Dict, List, Optional, Any, Set
import logging
import re
from pathlib import Path
import json

from ...domain.entities.explanation import (
    Language, Audience, ExplanationDepth, EducationalContent,
    ExplanationRequest, Reference
)
from ...domain.entities.antipattern_analysis import AntipatternType
from .exceptions import ContentGenerationError


logger = logging.getLogger(__name__)


class EducationalContentGenerator:
    """
    Generates educational content about code quality topics.
    
    This class is responsible for creating educational content to help users
    understand code quality issues, antipatterns, and best practices.
    """
    
    def __init__(self, resources_dir: Optional[Path] = None):
        """
        Initialize the educational content generator.
        
        Args:
            resources_dir: Directory containing educational resources
        """
        self.resources_dir = resources_dir
        self.educational_resources = self._load_educational_resources()
        self.code_examples = self._load_code_examples()
        self.references = self._load_references()
        
    def _load_educational_resources(self) -> Dict[str, Dict[str, Any]]:
        """Load educational resources for different topics."""
        return {
            "sql_injection": {
                "title": {
                    "en": "Understanding SQL Injection",
                    "es": "Entendiendo la Inyección SQL"
                },
                "description": {
                    "en": "SQL injection is a code injection technique that exploits vulnerabilities " +
                          "in the interface between web applications and database servers. " +
                          "It occurs when untrusted data is sent to an interpreter as part of a command or query.",
                    "es": "La inyección SQL es una técnica de inyección de código que explota vulnerabilidades " +
                          "en la interfaz entre aplicaciones web y servidores de bases de datos. " +
                          "Ocurre cuando datos no confiables se envían a un intérprete como parte de un comando o consulta."
                },
                "impact": {
                    "en": "Attackers can potentially read, modify, or delete database data, " +
                          "bypass authentication, or even execute administrative operations on the database.",
                    "es": "Los atacantes pueden potencialmente leer, modificar o eliminar datos de la base de datos, " +
                          "eludir la autenticación o incluso ejecutar operaciones administrativas en la base de datos."
                },
                "solution": {
                    "en": "Use parameterized queries (prepared statements) instead of concatenating user input. " +
                          "This ensures that user input is treated as data, not executable code.",
                    "es": "Use consultas parametrizadas (prepared statements) en lugar de concatenar la entrada del usuario. " +
                          "Esto asegura que la entrada del usuario se trate como datos, no como código ejecutable."
                },
                "learning_resources": [
                    {"title": "OWASP SQL Injection Prevention Cheat Sheet", "url": "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html"},
                    {"title": "Parameterized Queries", "url": "https://www.w3schools.com/sql/sql_injection.asp"},
                    {"title": "SQL Injection Explained", "url": "https://portswigger.net/web-security/sql-injection"}
                ]
            },
            "god_object": {
                "title": {
                    "en": "Understanding the God Object Antipattern",
                    "es": "Entendiendo el Antipatrón de Objeto Dios"
                },
                "description": {
                    "en": "A God Object is a class that knows too much or does too much. " +
                          "It's a class that has grown to do everything, violating the Single Responsibility Principle.",
                    "es": "Un Objeto Dios es una clase que sabe demasiado o hace demasiado. " +
                          "Es una clase que ha crecido para hacerlo todo, violando el Principio de Responsabilidad Única."
                },
                "impact": {
                    "en": "God Objects make code hard to maintain, test, and reuse. " +
                          "Changes to one aspect can affect unrelated functionality, leading to unexpected bugs.",
                    "es": "Los Objetos Dios hacen que el código sea difícil de mantener, probar y reutilizar. " +
                          "Los cambios en un aspecto pueden afectar funcionalidades no relacionadas, provocando errores inesperados."
                },
                "solution": {
                    "en": "Refactor by extracting cohesive groups of methods and fields into separate classes. " +
                          "Apply the Single Responsibility Principle to ensure each class has only one reason to change.",
                    "es": "Refactorice extrayendo grupos cohesivos de métodos y campos en clases separadas. " +
                          "Aplique el Principio de Responsabilidad Única para garantizar que cada clase tenga solo una razón para cambiar."
                },
                "learning_resources": [
                    {"title": "Single Responsibility Principle", "url": "https://en.wikipedia.org/wiki/Single-responsibility_principle"},
                    {"title": "Refactoring: Improving the Design of Existing Code", "url": "https://martinfowler.com/books/refactoring.html"},
                    {"title": "God Object Anti-Pattern", "url": "https://sourcemaking.com/antipatterns/god-class"}
                ]
            },
            "n_plus_one_query": {
                "title": {
                    "en": "Understanding the N+1 Query Problem",
                    "es": "Entendiendo el Problema de Consulta N+1"
                },
                "description": {
                    "en": "The N+1 query problem occurs when code needs to load a collection of items and a property " +
                          "for each item in that collection, resulting in N+1 database queries (1 for the collection, N for the properties).",
                    "es": "El problema de consulta N+1 ocurre cuando el código necesita cargar una colección de elementos y una propiedad " +
                          "para cada elemento en esa colección, resultando en N+1 consultas a la base de datos (1 para la colección, N para las propiedades)."
                },
                "impact": {
                    "en": "N+1 queries cause significant performance issues as the size of the collection grows, " +
                          "leading to slow page loads, increased database load, and poor user experience.",
                    "es": "Las consultas N+1 causan problemas significativos de rendimiento a medida que crece el tamaño de la colección, " +
                          "lo que lleva a cargas de página lentas, mayor carga en la base de datos y mala experiencia de usuario."
                },
                "solution": {
                    "en": "Use eager loading techniques like JOIN queries or batch loading to fetch all required data " +
                          "in a single query or a small number of queries.",
                    "es": "Utilice técnicas de carga anticipada como consultas JOIN o carga por lotes para obtener todos los datos necesarios " +
                          "en una sola consulta o un pequeño número de consultas."
                },
                "learning_resources": [
                    {"title": "What is the N+1 query problem?", "url": "https://stackoverflow.com/questions/97197/what-is-the-n1-selects-problem-in-orm-object-relational-mapping"},
                    {"title": "Eager Loading Explained", "url": "https://www.sitepoint.com/eager-loading-with-laravel-relationships/"},
                    {"title": "Solving N+1 Query Problem", "url": "https://medium.com/@marco.botto/the-n-1-problem-in-graphql-dd4921cb3c1a"}
                ]
            }
        }
    
    def _load_code_examples(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Load code examples for different topics and languages."""
        return {
            "sql_injection": {
                "python": {
                    "vulnerable": """
# Vulnerable code with SQL injection
user_id = request.GET.get('user_id')
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(query)
""",
                    "fixed": """
# Fixed code using parameterized query
user_id = request.GET.get('user_id')
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, [user_id])
"""
                },
                "javascript": {
                    "vulnerable": """
// Vulnerable code with SQL injection
const userId = req.query.userId;
const query = `SELECT * FROM users WHERE id = ${userId}`;
db.query(query);
""",
                    "fixed": """
// Fixed code using parameterized query
const userId = req.query.userId;
const query = `SELECT * FROM users WHERE id = ?`;
db.query(query, [userId]);
"""
                }
            },
            "god_object": {
                "python": {
                    "vulnerable": """
# God Object example
class UserManager:
    def __init__(self):
        self.db = Database()
        self.logger = Logger()
        self.email_service = EmailService()
        self.payment_gateway = PaymentGateway()
        
    def create_user(self, user_data):
        # User creation logic
        pass
        
    def update_user(self, user_id, user_data):
        # User update logic
        pass
        
    def delete_user(self, user_id):
        # User deletion logic
        pass
        
    def send_welcome_email(self, user_id):
        # Email sending logic
        pass
        
    def process_payment(self, user_id, amount):
        # Payment processing logic
        pass
        
    def generate_report(self, user_id):
        # Report generation logic
        pass
        
    def log_user_activity(self, user_id, activity):
        # Logging logic
        pass
""",
                    "fixed": """
# Refactored with Single Responsibility Principle
class UserRepository:
    def __init__(self):
        self.db = Database()
        
    def create_user(self, user_data):
        # User creation logic
        pass
        
    def update_user(self, user_id, user_data):
        # User update logic
        pass
        
    def delete_user(self, user_id):
        # User deletion logic
        pass

class EmailService:
    def send_welcome_email(self, user):
        # Email sending logic
        pass

class PaymentService:
    def __init__(self):
        self.payment_gateway = PaymentGateway()
        
    def process_payment(self, user_id, amount):
        # Payment processing logic
        pass

class ReportGenerator:
    def generate_report(self, user_id):
        # Report generation logic
        pass

class ActivityLogger:
    def __init__(self):
        self.logger = Logger()
        
    def log_user_activity(self, user_id, activity):
        # Logging logic
        pass
"""
                }
            },
            "n_plus_one_query": {
                "python": {
                    "vulnerable": """
# N+1 Query Problem
users = User.objects.all()  # 1 query to get all users

# For each user, a separate query is executed to get their posts
for user in users:
    posts = user.posts.all()  # N queries, one for each user
    for post in posts:
        print(f"{user.name}: {post.title}")
""",
                    "fixed": """
# Fixed with eager loading
users = User.objects.prefetch_related('posts').all()  # 1 query for users + 1 query for all posts

# No additional queries needed
for user in users:
    for post in user.posts.all():  # No additional queries
        print(f"{user.name}: {post.title}")
"""
                },
                "javascript": {
                    "vulnerable": """
// N+1 Query Problem
const users = await User.findAll();  // 1 query to get all users

// For each user, a separate query is executed to get their posts
for (const user of users) {
    const posts = await Post.findAll({ where: { userId: user.id } });  // N queries
    for (const post of posts) {
        console.log(`${user.name}: ${post.title}`);
    }
}
""",
                    "fixed": """
// Fixed with eager loading
const users = await User.findAll({
    include: [{
        model: Post,
        as: 'posts'
    }]
});  // 1 query for users with their posts

// No additional queries needed
for (const user of users) {
    for (const post of user.posts) {  // No additional queries
        console.log(`${user.name}: ${post.title}`);
    }
}
"""
                }
            }
        }
    
    def _load_references(self) -> Dict[str, List[Reference]]:
        """Load references for different topics."""
        return {
            "sql_injection": [
                Reference(
                    title="OWASP SQL Injection Prevention Cheat Sheet",
                    url="https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html",
                    description="Comprehensive guide for preventing SQL injection vulnerabilities",
                    type="documentation"
                ),
                Reference(
                    title="CWE-89: SQL Injection",
                    url="https://cwe.mitre.org/data/definitions/89.html",
                    description="Common Weakness Enumeration entry for SQL injection",
                    type="documentation"
                ),
                Reference(
                    title="SQL Injection Attacks and Defense",
                    url="https://www.amazon.com/SQL-Injection-Attacks-Defense-Second/dp/1597499633",
                    description="Book on SQL injection attacks and defense strategies",
                    type="book"
                )
            ],
            "god_object": [
                Reference(
                    title="Refactoring: Improving the Design of Existing Code",
                    url="https://martinfowler.com/books/refactoring.html",
                    description="Martin Fowler's book on refactoring techniques",
                    type="book"
                ),
                Reference(
                    title="Clean Code: A Handbook of Agile Software Craftsmanship",
                    url="https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882",
                    description="Robert C. Martin's book on writing clean, maintainable code",
                    type="book"
                ),
                Reference(
                    title="The God Object Antipattern",
                    url="https://sourcemaking.com/antipatterns/god-class",
                    description="Detailed explanation of the God Object antipattern",
                    type="web"
                )
            ],
            "n_plus_one_query": [
                Reference(
                    title="N+1 Query Problem",
                    url="https://stackoverflow.com/questions/97197/what-is-the-n1-selects-problem-in-orm-object-relational-mapping",
                    description="Stack Overflow discussion on the N+1 query problem",
                    type="web"
                ),
                Reference(
                    title="Eager Loading Explained",
                    url="https://www.sitepoint.com/eager-loading-with-laravel-relationships/",
                    description="Guide to eager loading in ORMs",
                    type="web"
                ),
                Reference(
                    title="Database Performance Antipatterns",
                    url="https://www.red-gate.com/simple-talk/sql/performance/database-performance-antipatterns/",
                    description="Common database performance antipatterns and solutions",
                    type="web"
                )
            ]
        }
    
    async def generate_educational_content(
        self,
        analysis_result: Any,
        request: ExplanationRequest
    ) -> List[EducationalContent]:
        """
        Generate educational content based on analysis results.
        
        Args:
            analysis_result: The analysis results
            request: The explanation request
            
        Returns:
            A list of educational content items
        """
        try:
            educational_content = []
            
            # Identify topics to generate content for
            topics = await self._identify_relevant_topics(analysis_result)
            
            # Generate content for each topic
            for topic in topics:
                content = await self._generate_content_for_topic(
                    topic, request.language, request.audience
                )
                if content:
                    educational_content.append(content)
            
            return educational_content
            
        except Exception as e:
            logger.error(f"Error generating educational content: {str(e)}")
            raise ContentGenerationError(f"Failed to generate educational content: {str(e)}")
    
    async def _identify_relevant_topics(self, analysis_result: Any) -> List[str]:
        """
        Identify relevant topics based on analysis results.
        
        Args:
            analysis_result: The analysis results
            
        Returns:
            A list of relevant topics
        """
        topics = set()
        
        # Check for antipatterns
        antipatterns = getattr(analysis_result, 'antipatterns', [])
        for antipattern in antipatterns:
            pattern_type = getattr(antipattern, 'pattern_type', None)
            if pattern_type:
                if isinstance(pattern_type, AntipatternType):
                    pattern_type = pattern_type.value
                
                # Convert to topic key
                topic_key = self._convert_to_topic_key(pattern_type)
                if topic_key in self.educational_resources:
                    topics.add(topic_key)
        
        # Check for issues
        issues = getattr(analysis_result, 'issues', [])
        for issue in issues:
            issue_type = getattr(issue, 'issue_type', None)
            if issue_type:
                # Convert to topic key
                topic_key = self._convert_to_topic_key(issue_type)
                if topic_key in self.educational_resources:
                    topics.add(topic_key)
        
        return list(topics)
    
    def _convert_to_topic_key(self, name: str) -> str:
        """Convert a name to a topic key."""
        # Convert to lowercase and replace spaces with underscores
        key = name.lower().replace(' ', '_')
        
        # Map common variations
        mapping = {
            'sql_injection': 'sql_injection',
            'god_object': 'god_object',
            'god_class': 'god_object',
            'long_method': 'long_method',
            'n+1_query': 'n_plus_one_query',
            'n_plus_one_query': 'n_plus_one_query'
        }
        
        return mapping.get(key, key)
    
    async def _generate_content_for_topic(
        self,
        topic: str,
        language: Language,
        audience: Audience
    ) -> Optional[EducationalContent]:
        """
        Generate educational content for a specific topic.
        
        Args:
            topic: The topic to generate content for
            language: The target language
            audience: The target audience
            
        Returns:
            Educational content for the topic
        """
        if topic not in self.educational_resources:
            return None
            
        resource = self.educational_resources[topic]
        
        # Get language-specific content
        lang_code = 'es' if language == Language.SPANISH else 'en'
        
        title = resource['title'].get(lang_code, resource['title'].get('en', topic))
        description = resource['description'].get(lang_code, resource['description'].get('en', ''))
        impact = resource['impact'].get(lang_code, resource['impact'].get('en', ''))
        solution = resource['solution'].get(lang_code, resource['solution'].get('en', ''))
        
        # Get code examples
        code_examples_text = await self._format_code_examples(topic, lang_code)
        
        # Format learning resources
        learning_resources_text = await self._format_learning_resources(
            resource.get('learning_resources', []), lang_code
        )
        
        # Combine all content
        content_text = f"{description}\n\n"
        content_text += f"**{self._translate('Impact', lang_code)}:**\n{impact}\n\n"
        content_text += f"**{self._translate('Solution', lang_code)}:**\n{solution}\n\n"
        
        if code_examples_text:
            content_text += f"**{self._translate('Code Examples', lang_code)}:**\n{code_examples_text}\n\n"
            
        if learning_resources_text:
            content_text += f"**{self._translate('Learning Resources', lang_code)}:**\n{learning_resources_text}\n\n"
        
        # Adapt content for audience
        content_text = await self._adapt_content_for_audience(content_text, audience)
        
        # Get references
        references = self.references.get(topic, [])
        
        return EducationalContent(
            title=title,
            content=content_text,
            content_type="text",
            tags=[topic],
            references=[ref.title for ref in references]
        )
    
    async def _format_code_examples(self, topic: str, lang_code: str) -> str:
        """Format code examples for a topic."""
        if topic not in self.code_examples:
            return ""
            
        examples_text = ""
        
        # Get examples for Python (default)
        python_examples = self.code_examples[topic].get('python', {})
        if python_examples:
            examples_text += f"**Python:**\n\n"
            examples_text += f"*{self._translate('Vulnerable Code', lang_code)}:*\n"
            examples_text += f"```python\n{python_examples.get('vulnerable', '')}\n```\n\n"
            examples_text += f"*{self._translate('Fixed Code', lang_code)}:*\n"
            examples_text += f"```python\n{python_examples.get('fixed', '')}\n```\n\n"
        
        # Get examples for JavaScript if available
        js_examples = self.code_examples[topic].get('javascript', {})
        if js_examples:
            examples_text += f"**JavaScript:**\n\n"
            examples_text += f"*{self._translate('Vulnerable Code', lang_code)}:*\n"
            examples_text += f"```javascript\n{js_examples.get('vulnerable', '')}\n```\n\n"
            examples_text += f"*{self._translate('Fixed Code', lang_code)}:*\n"
            examples_text += f"```javascript\n{js_examples.get('fixed', '')}\n```\n\n"
        
        return examples_text
    
    async def _format_learning_resources(
        self,
        resources: List[Dict[str, str]],
        lang_code: str
    ) -> str:
        """Format learning resources."""
        if not resources:
            return ""
            
        resources_text = ""
        
        for resource in resources:
            title = resource.get('title', '')
            url = resource.get('url', '')
            
            if title and url:
                resources_text += f"- [{title}]({url})\n"
        
        return resources_text
    
    def _translate(self, text: str, lang_code: str) -> str:
        """Translate common terms."""
        translations = {
            'Impact': {'en': 'Impact', 'es': 'Impacto'},
            'Solution': {'en': 'Solution', 'es': 'Solución'},
            'Code Examples': {'en': 'Code Examples', 'es': 'Ejemplos de Código'},
            'Learning Resources': {'en': 'Learning Resources', 'es': 'Recursos de Aprendizaje'},
            'Vulnerable Code': {'en': 'Vulnerable Code', 'es': 'Código Vulnerable'},
            'Fixed Code': {'en': 'Fixed Code', 'es': 'Código Corregido'}
        }
        
        if text in translations and lang_code in translations[text]:
            return translations[text][lang_code]
        
        return text
    
    async def _adapt_content_for_audience(
        self,
        content: str,
        audience: Audience
    ) -> str:
        """
        Adapt content for different audiences.
        
        Args:
            content: The content to adapt
            audience: The target audience
            
        Returns:
            The adapted content
        """
        if audience == Audience.JUNIOR_DEVELOPER:
            # Add more explanations for junior developers
            content = content.replace(
                "parameterized queries",
                "parameterized queries (special database queries that separate code from data)"
            )
            content = content.replace(
                "Single Responsibility Principle",
                "Single Responsibility Principle (a class should have only one reason to change)"
            )
            content = content.replace(
                "eager loading",
                "eager loading (loading related data in advance)"
            )
            
        elif audience == Audience.PROJECT_MANAGER:
            # Focus on business impact for project managers
            content += "\n\n**Business Impact:**\n"
            if "SQL injection" in content:
                content += "SQL injection vulnerabilities can lead to data breaches, which may result in regulatory fines, legal liabilities, and damage to company reputation. Fixing these issues early is much less costly than dealing with a security breach.\n"
            elif "God Object" in content:
                content += "God Objects increase maintenance costs and slow down development as the codebase grows. Refactoring them improves developer productivity and reduces bugs, leading to faster delivery and lower costs in the long term.\n"
            elif "N+1 query" in content:
                content += "N+1 query issues can significantly impact application performance and user experience, especially as data volumes grow. Fixing these issues improves application scalability and reduces infrastructure costs.\n"
            
        elif audience == Audience.SECURITY_TEAM:
            # Add security-specific details for security teams
            if "SQL injection" in content:
                content += "\n\n**Security Considerations:**\n"
                content += "SQL injection is among the OWASP Top 10 vulnerabilities and can lead to complete database compromise. Ensure all database access points use parameterized queries and implement proper input validation. Consider implementing a web application firewall (WAF) as an additional layer of defense.\n"
        
        return content
    
    async def get_educational_content_for_topic(
        self,
        topic: str,
        language: Language,
        audience: Audience
    ) -> Optional[EducationalContent]:
        """
        Get educational content for a specific topic.
        
        Args:
            topic: The topic to get content for
            language: The target language
            audience: The target audience
            
        Returns:
            Educational content for the topic
        """
        topic_key = self._convert_to_topic_key(topic)
        return await self._generate_content_for_topic(topic_key, language, audience)
    
    async def get_references_for_topic(
        self,
        topic: str
    ) -> List[Reference]:
        """
        Get references for a specific topic.
        
        Args:
            topic: The topic to get references for
            
        Returns:
            A list of references for the topic
        """
        topic_key = self._convert_to_topic_key(topic)
        return self.references.get(topic_key, [])
a