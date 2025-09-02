"""
Value Objects para lenguajes de programación.

Este módulo define los tipos de lenguajes de programación soportados
por el sistema de análisis de código.
"""

from enum import Enum
from typing import List, Optional


class ProgrammingLanguage(Enum):
    """Enumeración de lenguajes de programación soportados."""
    
    # Lenguajes principales soportados por Tree-sitter
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    RUST = "rust"
    JAVA = "java"
    GO = "go"
    CPP = "cpp"
    CSHARP = "csharp"
    
    # Lenguajes adicionales
    C = "c"
    SCALA = "scala"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    PHP = "php"
    RUBY = "ruby"
    PERL = "perl"
    BASH = "bash"
    POWERSHELL = "powershell"
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    HTML = "html"
    CSS = "css"
    MARKDOWN = "markdown"
    
    # Lenguajes de configuración
    TOML = "toml"
    INI = "ini"
    ENV = "env"
    
    # Lenguajes de base de datos
    SQL = "sql"
    PLSQL = "plsql"
    TSQL = "tsql"
    
    # Lenguajes de infraestructura
    DOCKERFILE = "dockerfile"
    KUBERNETES = "kubernetes"
    TERRAFORM = "terraform"
    ANSIBLE = "ansible"
    
    # Lenguajes desconocidos o no soportados
    UNKNOWN = "unknown"
    
    @classmethod
    def from_extension(cls, extension: str) -> Optional['ProgrammingLanguage']:
        """Obtiene el lenguaje de programación basado en la extensión del archivo."""
        extension_map = {
            # Python
            'py': cls.PYTHON,
            'pyw': cls.PYTHON,
            'pyi': cls.PYTHON,
            'pyx': cls.PYTHON,
            'pxd': cls.PYTHON,
            
            # TypeScript
            'ts': cls.TYPESCRIPT,
            'tsx': cls.TYPESCRIPT,
            
            # JavaScript
            'js': cls.JAVASCRIPT,
            'jsx': cls.JAVASCRIPT,
            'mjs': cls.JAVASCRIPT,
            'cjs': cls.JAVASCRIPT,
            
            # Rust
            'rs': cls.RUST,
            
            # Java
            'java': cls.JAVA,
            'class': cls.JAVA,
            'jar': cls.JAVA,
            
            # Go
            'go': cls.GO,
            
            # C++
            'cpp': cls.CPP,
            'cc': cls.CPP,
            'cxx': cls.CPP,
            'hpp': cls.CPP,
            
            # C#
            'cs': cls.CSHARP,
            
            # C
            'c': cls.C,
            'h': cls.C,  # Los archivos .h pueden ser tanto C como C++, pero por defecto C
            
            # Scala
            'scala': cls.SCALA,
            'sc': cls.SCALA,
            
            # Kotlin
            'kt': cls.KOTLIN,
            'kts': cls.KOTLIN,
            
            # Swift
            'swift': cls.SWIFT,
            
            # PHP
            'php': cls.PHP,
            'phtml': cls.PHP,
            
            # Ruby
            'rb': cls.RUBY,
            'erb': cls.RUBY,
            
            # Perl
            'pl': cls.PERL,
            'pm': cls.PERL,
            
            # Shell
            'sh': cls.BASH,
            'bash': cls.BASH,
            'zsh': cls.BASH,
            'fish': cls.BASH,
            'ps1': cls.POWERSHELL,
            
            # Markup y configuración
            'yaml': cls.YAML,
            'yml': cls.YAML,
            'json': cls.JSON,
            'xml': cls.XML,
            'html': cls.HTML,
            'htm': cls.HTML,
            'css': cls.CSS,
            'scss': cls.CSS,
            'sass': cls.CSS,
            'less': cls.CSS,
            'md': cls.MARKDOWN,
            'markdown': cls.MARKDOWN,
            
            # Configuración
            'toml': cls.TOML,
            'ini': cls.INI,
            'env': cls.ENV,
            
            # Base de datos
            'sql': cls.SQL,
            'plsql': cls.PLSQL,
            'tsql': cls.TSQL,
            
            # Infraestructura
            'dockerfile': cls.DOCKERFILE,
            'yaml': cls.KUBERNETES,
            'tf': cls.TERRAFORM,
            'tfvars': cls.TERRAFORM,
            'yml': cls.ANSIBLE,
        }
        
        return extension_map.get(extension.lower())
    
    @classmethod
    def from_filename(cls, filename: str) -> Optional['ProgrammingLanguage']:
        """Obtiene el lenguaje de programación basado en el nombre del archivo."""
        filename_map = {
            # Archivos de configuración específicos
            'Cargo.toml': cls.RUST,
            'Cargo.lock': cls.RUST,
            'package.json': cls.JAVASCRIPT,
            'package-lock.json': cls.JAVASCRIPT,
            'yarn.lock': cls.JAVASCRIPT,
            'tsconfig.json': cls.TYPESCRIPT,
            'go.mod': cls.GO,
            'go.sum': cls.GO,
            'pom.xml': cls.JAVA,
            'build.gradle': cls.JAVA,
            'requirements.txt': cls.PYTHON,
            'Pipfile': cls.PYTHON,
            'poetry.lock': cls.PYTHON,
            'setup.py': cls.PYTHON,
            'setup.cfg': cls.PYTHON,
            'pyproject.toml': cls.PYTHON,
            'Dockerfile': cls.DOCKERFILE,
            'docker-compose.yml': cls.DOCKERFILE,
            'docker-compose.yaml': cls.DOCKERFILE,
            'Makefile': cls.BASH,
            'CMakeLists.txt': cls.CPP,
            '.gitignore': cls.UNKNOWN,
            '.gitattributes': cls.UNKNOWN,
            'README.md': cls.MARKDOWN,
            'CHANGELOG.md': cls.MARKDOWN,
            'LICENSE': cls.UNKNOWN,
        }
        
        return filename_map.get(filename)
    
    @classmethod
    def get_supported_languages(cls) -> List['ProgrammingLanguage']:
        """Obtiene la lista de lenguajes completamente soportados."""
        return [
            cls.PYTHON,
            cls.TYPESCRIPT,
            cls.JAVASCRIPT,
            cls.RUST,
            cls.JAVA,
            cls.GO,
            cls.CPP,
            cls.CSHARP,
        ]
    
    @classmethod
    def get_experimental_languages(cls) -> List['ProgrammingLanguage']:
        """Obtiene la lista de lenguajes en fase experimental."""
        return [
            cls.C,
            cls.SCALA,
            cls.KOTLIN,
            cls.SWIFT,
            cls.PHP,
            cls.RUBY,
            cls.PERL,
        ]
    
    @classmethod
    def get_config_languages(cls) -> List['ProgrammingLanguage']:
        """Obtiene la lista de lenguajes de configuración."""
        return [
            cls.YAML,
            cls.JSON,
            cls.TOML,
            cls.INI,
            cls.ENV,
            cls.XML,
        ]
    
    @classmethod
    def get_markup_languages(cls) -> List['ProgrammingLanguage']:
        """Obtiene la lista de lenguajes de markup."""
        return [
            cls.HTML,
            cls.CSS,
            cls.MARKDOWN,
        ]
    
    @classmethod
    def get_infrastructure_languages(cls) -> List['ProgrammingLanguage']:
        """Obtiene la lista de lenguajes de infraestructura."""
        return [
            cls.DOCKERFILE,
            cls.KUBERNETES,
            cls.TERRAFORM,
            cls.ANSIBLE,
            cls.SQL,
            cls.PLSQL,
            cls.TSQL,
        ]
    
    def get_name(self) -> str:
        """Obtiene el nombre legible del lenguaje."""
        name_map = {
            self.PYTHON: "Python",
            self.TYPESCRIPT: "TypeScript",
            self.JAVASCRIPT: "JavaScript",
            self.RUST: "Rust",
            self.JAVA: "Java",
            self.GO: "Go",
            self.CPP: "C++",
            self.CSHARP: "C#",
            self.C: "C",
            self.SCALA: "Scala",
            self.KOTLIN: "Kotlin",
            self.SWIFT: "Swift",
            self.PHP: "PHP",
            self.RUBY: "Ruby",
            self.PERL: "Perl",
            self.BASH: "Bash",
            self.POWERSHELL: "PowerShell",
            self.YAML: "YAML",
            self.JSON: "JSON",
            self.XML: "XML",
            self.HTML: "HTML",
            self.CSS: "CSS",
            self.MARKDOWN: "Markdown",
            self.TOML: "TOML",
            self.INI: "INI",
            self.ENV: "Environment",
            self.SQL: "SQL",
            self.PLSQL: "PL/SQL",
            self.TSQL: "T-SQL",
            self.DOCKERFILE: "Dockerfile",
            self.KUBERNETES: "Kubernetes",
            self.TERRAFORM: "Terraform",
            self.ANSIBLE: "Ansible",
            self.UNKNOWN: "Unknown",
        }
        
        return name_map.get(self, self.value.capitalize())
    
    def get_file_extensions(self) -> List[str]:
        """Obtiene las extensiones de archivo asociadas con este lenguaje."""
        extension_map = {
            self.PYTHON: ['py', 'pyw', 'pyi', 'pyx', 'pxd'],
            self.TYPESCRIPT: ['ts', 'tsx'],
            self.JAVASCRIPT: ['js', 'jsx', 'mjs', 'cjs'],
            self.RUST: ['rs'],
            self.JAVA: ['java', 'class', 'jar'],
            self.GO: ['go'],
            self.CPP: ['cpp', 'cc', 'cxx', 'hpp', 'h'],
            self.CSHARP: ['cs'],
            self.C: ['c', 'h'],
            self.SCALA: ['scala', 'sc'],
            self.KOTLIN: ['kt', 'kts'],
            self.SWIFT: ['swift'],
            self.PHP: ['php', 'phtml'],
            self.RUBY: ['rb', 'erb'],
            self.PERL: ['pl', 'pm'],
            self.BASH: ['sh', 'bash', 'zsh', 'fish'],
            self.POWERSHELL: ['ps1'],
            self.YAML: ['yaml', 'yml'],
            self.JSON: ['json'],
            self.XML: ['xml'],
            self.HTML: ['html', 'htm'],
            self.CSS: ['css', 'scss', 'sass', 'less'],
            self.MARKDOWN: ['md', 'markdown'],
            self.TOML: ['toml'],
            self.INI: ['ini'],
            self.ENV: ['env'],
            self.SQL: ['sql'],
            self.PLSQL: ['plsql'],
            self.TSQL: ['tsql'],
            self.DOCKERFILE: ['dockerfile'],
            self.KUBERNETES: ['yaml', 'yml'],
            self.TERRAFORM: ['tf', 'tfvars'],
            self.ANSIBLE: ['yml'],
            self.UNKNOWN: [],
        }
        
        return extension_map.get(self, [])
    
    def has_extension(self, extension: str) -> bool:
        """Verifica si el lenguaje tiene la extensión especificada."""
        return extension.lower() in [ext.lower() for ext in self.get_file_extensions()]
    
    def is_supported(self) -> bool:
        """Verifica si el lenguaje está completamente soportado."""
        return self in self.get_supported_languages()
    
    def is_experimental(self) -> bool:
        """Verifica si el lenguaje está en fase experimental."""
        return self in self.get_experimental_languages()
    
    def is_config(self) -> bool:
        """Verifica si el lenguaje es de configuración."""
        return self in self.get_config_languages()
    
    def is_markup(self) -> bool:
        """Verifica si el lenguaje es de markup."""
        return self in self.get_markup_languages()
    
    def is_infrastructure(self) -> bool:
        """Verifica si el lenguaje es de infraestructura."""
        return self in self.get_infrastructure_languages()
    
    def __str__(self) -> str:
        """Representación string del lenguaje."""
        return self.get_name()
    
    def __repr__(self) -> str:
        """Representación de debug del lenguaje."""
        return f"ProgrammingLanguage.{self.name}"
