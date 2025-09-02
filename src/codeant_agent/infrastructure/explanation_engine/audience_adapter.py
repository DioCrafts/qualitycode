"""
Audience Adapter for the Explanation Engine.

This module is responsible for adapting explanations to different audiences,
such as junior developers, senior developers, project managers, etc.
"""
from typing import Dict, List, Optional, Any, Set
import logging
import re

from ...domain.entities.explanation import (
    Audience, ExplanationDepth, PersonalizationContext
)
from .exceptions import AudienceAdapterError, UnsupportedAudienceError


logger = logging.getLogger(__name__)


class AudienceAdapter:
    """
    Adapts explanations to different audiences.
    
    This class provides functionality to adapt explanations to different
    audiences, adjusting the level of detail, technical depth, and focus
    based on the target audience.
    """
    
    def __init__(self):
        """Initialize the audience adapter."""
        self.audience_profiles = self._load_audience_profiles()
        self.adaptation_strategies = self._load_adaptation_strategies()
        
    def _load_audience_profiles(self) -> Dict[Audience, Dict[str, Any]]:
        """Load profiles for different audiences."""
        return {
            Audience.JUNIOR_DEVELOPER: {
                "technical_depth": 0.7,  # 0.0 = non-technical, 1.0 = highly technical
                "detail_level": 0.8,     # 0.0 = brief, 1.0 = comprehensive
                "focus_areas": ["educational", "examples", "best_practices"],
                "terminology_level": "basic",
                "assumed_knowledge": ["basic_programming", "basic_tools"],
                "content_preferences": {
                    "include_examples": True,
                    "include_explanations": True,
                    "include_background": True,
                    "include_alternatives": False
                }
            },
            Audience.SENIOR_DEVELOPER: {
                "technical_depth": 0.9,
                "detail_level": 0.7,
                "focus_areas": ["technical_details", "performance", "architecture"],
                "terminology_level": "advanced",
                "assumed_knowledge": ["advanced_programming", "design_patterns", "algorithms"],
                "content_preferences": {
                    "include_examples": True,
                    "include_explanations": False,
                    "include_background": False,
                    "include_alternatives": True
                }
            },
            Audience.TECHNICAL_LEAD: {
                "technical_depth": 0.8,
                "detail_level": 0.6,
                "focus_areas": ["architecture", "technical_debt", "team_impact"],
                "terminology_level": "advanced",
                "assumed_knowledge": ["system_design", "project_management", "team_coordination"],
                "content_preferences": {
                    "include_examples": False,
                    "include_explanations": False,
                    "include_background": False,
                    "include_alternatives": True
                }
            },
            Audience.SOFTWARE_ARCHITECT: {
                "technical_depth": 0.9,
                "detail_level": 0.7,
                "focus_areas": ["architecture", "scalability", "maintainability"],
                "terminology_level": "expert",
                "assumed_knowledge": ["system_design", "enterprise_architecture", "patterns"],
                "content_preferences": {
                    "include_examples": False,
                    "include_explanations": False,
                    "include_background": False,
                    "include_alternatives": True
                }
            },
            Audience.PROJECT_MANAGER: {
                "technical_depth": 0.4,
                "detail_level": 0.5,
                "focus_areas": ["timeline_impact", "resource_needs", "business_impact"],
                "terminology_level": "basic",
                "assumed_knowledge": ["project_management", "basic_development_process"],
                "content_preferences": {
                    "include_examples": False,
                    "include_explanations": True,
                    "include_background": True,
                    "include_alternatives": False
                }
            },
            Audience.QUALITY_ASSURANCE: {
                "technical_depth": 0.6,
                "detail_level": 0.8,
                "focus_areas": ["testability", "edge_cases", "regression_risks"],
                "terminology_level": "intermediate",
                "assumed_knowledge": ["testing_methodologies", "quality_metrics"],
                "content_preferences": {
                    "include_examples": True,
                    "include_explanations": True,
                    "include_background": False,
                    "include_alternatives": False
                }
            },
            Audience.SECURITY_TEAM: {
                "technical_depth": 0.8,
                "detail_level": 0.9,
                "focus_areas": ["security_vulnerabilities", "attack_vectors", "compliance"],
                "terminology_level": "advanced",
                "assumed_knowledge": ["security_principles", "common_vulnerabilities"],
                "content_preferences": {
                    "include_examples": True,
                    "include_explanations": False,
                    "include_background": False,
                    "include_alternatives": True
                }
            },
            Audience.BUSINESS_STAKEHOLDER: {
                "technical_depth": 0.2,
                "detail_level": 0.3,
                "focus_areas": ["business_impact", "cost", "timeline"],
                "terminology_level": "non_technical",
                "assumed_knowledge": ["business_domain"],
                "content_preferences": {
                    "include_examples": False,
                    "include_explanations": True,
                    "include_background": True,
                    "include_alternatives": False
                }
            }
        }
    
    def _load_adaptation_strategies(self) -> Dict[str, Any]:
        """Load strategies for adapting content to different audiences."""
        return {
            "technical_depth": {
                "reduce": self._reduce_technical_depth,
                "increase": self._increase_technical_depth
            },
            "detail_level": {
                "reduce": self._reduce_detail_level,
                "increase": self._increase_detail_level
            },
            "terminology": {
                "simplify": self._simplify_terminology,
                "elevate": self._elevate_terminology
            },
            "focus": {
                "shift": self._shift_focus
            },
            "structure": {
                "reorganize": self._reorganize_content
            }
        }
    
    async def adapt_content(
        self,
        content: str,
        audience: Audience,
        personalization_context: Optional[PersonalizationContext] = None
    ) -> str:
        """
        Adapt content to the specified audience.
        
        Args:
            content: The content to adapt
            audience: The target audience
            personalization_context: Optional personalization context
            
        Returns:
            The adapted content for the target audience
        """
        if audience not in self.audience_profiles:
            raise UnsupportedAudienceError(f"Unsupported audience: {audience}")
            
        try:
            # Get audience profile
            profile = self.audience_profiles[audience]
            
            # Apply adaptation strategies
            adapted_content = content
            
            # Adjust technical depth
            target_technical_depth = profile["technical_depth"]
            adapted_content = await self._adjust_technical_depth(
                adapted_content, target_technical_depth
            )
            
            # Adjust detail level
            target_detail_level = profile["detail_level"]
            adapted_content = await self._adjust_detail_level(
                adapted_content, target_detail_level
            )
            
            # Adjust terminology
            terminology_level = profile["terminology_level"]
            adapted_content = await self._adjust_terminology(
                adapted_content, terminology_level
            )
            
            # Shift focus to relevant areas
            focus_areas = profile["focus_areas"]
            adapted_content = await self._adjust_focus(
                adapted_content, focus_areas
            )
            
            # Apply personalization if context is provided
            if personalization_context:
                adapted_content = await self._apply_personalization(
                    adapted_content, personalization_context
                )
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"Error adapting content for audience {audience}: {str(e)}")
            raise AudienceAdapterError(f"Failed to adapt content: {str(e)}")
    
    async def _adjust_technical_depth(
        self,
        content: str,
        target_depth: float
    ) -> str:
        """
        Adjust the technical depth of the content.
        
        Args:
            content: The content to adjust
            target_depth: The target technical depth (0.0-1.0)
            
        Returns:
            The adjusted content
        """
        # For now, we'll use a simple approach based on the target depth
        if target_depth < 0.4:
            # Non-technical audience
            return await self._reduce_technical_depth(content, 2)
        elif target_depth < 0.7:
            # Moderately technical audience
            return await self._reduce_technical_depth(content, 1)
        elif target_depth > 0.8:
            # Highly technical audience
            return await self._increase_technical_depth(content, 1)
        else:
            # Default technical level
            return content
    
    async def _reduce_technical_depth(self, content: str, level: int = 1) -> str:
        """
        Reduce the technical depth of the content.
        
        Args:
            content: The content to adjust
            level: The level of reduction (1=slight, 2=moderate, 3=significant)
            
        Returns:
            The adjusted content
        """
        # Replace technical terms with simpler explanations
        simplifications = {
            r'\bcyclomatic complexity\b': 'code complexity',
            r'\bmaintainability index\b': 'ease of maintenance',
            r'\bpolymorphism\b': 'flexible code structure',
            r'\binheritance\b': 'code reuse mechanism',
            r'\bencapsulation\b': 'data protection',
            r'\babstraction\b': 'simplified representation',
            r'\bsql injection\b': 'security vulnerability',
            r'\brefactoring\b': 'code improvement',
            r'\bdesign pattern\b': 'proven solution approach'
        }
        
        # Apply more aggressive simplification for higher levels
        if level >= 2:
            more_simplifications = {
                r'\balgorithm\b': 'procedure',
                r'\bfunction\b': 'code block',
                r'\bmethod\b': 'code block',
                r'\bparameter\b': 'input value',
                r'\bvariable\b': 'data container',
                r'\bobject\b': 'code entity',
                r'\bclass\b': 'code template',
                r'\binterface\b': 'connection point',
                r'\bimplementation\b': 'actual code',
                r'\binstance\b': 'specific version',
                r'\bcompilation\b': 'code processing',
                r'\bexecution\b': 'running the code',
                r'\bdebug\b': 'find and fix errors',
                r'\bstack trace\b': 'error location information',
                r'\bexception\b': 'error situation',
                r'\bmemory leak\b': 'resource waste',
                r'\bgarbage collection\b': 'automatic cleanup',
                r'\basynchronous\b': 'non-blocking',
                r'\bsynchronous\b': 'blocking',
                r'\bconcurrency\b': 'simultaneous operation',
                r'\bthreading\b': 'parallel processing',
                r'\brecursion\b': 'self-referencing process'
            }
            simplifications.update(more_simplifications)
        
        result = content
        for pattern, replacement in simplifications.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Remove or simplify code examples
        if level >= 2:
            # Replace code blocks with simplified descriptions
            result = re.sub(
                r'```(?:.*?)\n(.*?)```',
                r'[Code example removed for simplicity]',
                result,
                flags=re.DOTALL
            )
        
        # Add explanatory notes for technical concepts
        if level >= 1:
            # Add explanations for common technical terms
            explanations = {
                r'\bAPI\b': 'API (a way for different software to communicate)',
                r'\bJSON\b': 'JSON (a data format)',
                r'\bXML\b': 'XML (a structured data format)',
                r'\bHTTP\b': 'HTTP (the protocol used for web communication)',
                r'\bREST\b': 'REST (a web communication standard)',
                r'\bSOAP\b': 'SOAP (a messaging protocol)',
                r'\bSQL\b': 'SQL (database query language)',
                r'\bNoSQL\b': 'NoSQL (alternative database technology)',
                r'\bORM\b': 'ORM (a way to work with databases in code)'
            }
            
            for pattern, explanation in explanations.items():
                # Only replace the first occurrence to avoid repetition
                match = re.search(pattern, result, re.IGNORECASE)
                if match:
                    result = result[:match.start()] + explanation + result[match.end():]
        
        return result
    
    async def _increase_technical_depth(self, content: str, level: int = 1) -> str:
        """
        Increase the technical depth of the content.
        
        Args:
            content: The content to adjust
            level: The level of increase (1=slight, 2=moderate, 3=significant)
            
        Returns:
            The adjusted content
        """
        # Replace simplified terms with more technical terminology
        technicalizations = {
            r'\bcode complexity\b': 'cyclomatic complexity',
            r'\bease of maintenance\b': 'maintainability index',
            r'\bflexible code structure\b': 'polymorphism',
            r'\bcode reuse mechanism\b': 'inheritance',
            r'\bdata protection\b': 'encapsulation',
            r'\bsimplified representation\b': 'abstraction',
            r'\bsecurity vulnerability\b': 'security exploit vector',
            r'\bcode improvement\b': 'refactoring',
            r'\bproven solution approach\b': 'design pattern'
        }
        
        result = content
        for pattern, replacement in technicalizations.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        # Remove explanatory notes for technical terms
        if level >= 1:
            # Remove explanations for common technical terms
            result = re.sub(r'API \(a way for different software to communicate\)', 'API', result)
            result = re.sub(r'JSON \(a data format\)', 'JSON', result)
            result = re.sub(r'XML \(a structured data format\)', 'XML', result)
            result = re.sub(r'HTTP \(the protocol used for web communication\)', 'HTTP', result)
            result = re.sub(r'REST \(a web communication standard\)', 'REST', result)
            result = re.sub(r'SQL \(database query language\)', 'SQL', result)
            result = re.sub(r'ORM \(a way to work with databases in code\)', 'ORM', result)
        
        # Add more technical details if needed
        if level >= 2:
            # This would be more complex in a real implementation
            pass
        
        return result
    
    async def _adjust_detail_level(
        self,
        content: str,
        target_detail: float
    ) -> str:
        """
        Adjust the detail level of the content.
        
        Args:
            content: The content to adjust
            target_detail: The target detail level (0.0-1.0)
            
        Returns:
            The adjusted content
        """
        # For now, we'll use a simple approach based on the target detail level
        if target_detail < 0.4:
            # Brief content
            return await self._reduce_detail_level(content, 2)
        elif target_detail < 0.6:
            # Moderately detailed content
            return await self._reduce_detail_level(content, 1)
        elif target_detail > 0.8:
            # Highly detailed content
            return await self._increase_detail_level(content, 1)
        else:
            # Default detail level
            return content
    
    async def _reduce_detail_level(self, content: str, level: int = 1) -> str:
        """
        Reduce the detail level of the content.
        
        Args:
            content: The content to adjust
            level: The level of reduction (1=slight, 2=moderate, 3=significant)
            
        Returns:
            The adjusted content
        """
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        
        # If there's only one paragraph, we can't reduce further
        if len(paragraphs) <= 1:
            return content
        
        # For slight reduction, remove examples and detailed explanations
        if level == 1:
            # Remove paragraphs that appear to be examples
            filtered_paragraphs = []
            for paragraph in paragraphs:
                if not (re.search(r'(?:example|for instance|for example|e\.g\.)', paragraph, re.IGNORECASE) or
                        re.search(r'```', paragraph)):
                    filtered_paragraphs.append(paragraph)
            
            # If we removed too much, keep at least half of the original content
            if len(filtered_paragraphs) < len(paragraphs) // 2:
                filtered_paragraphs = paragraphs[:len(paragraphs) // 2]
                
            return '\n\n'.join(filtered_paragraphs)
            
        # For moderate reduction, keep only the most important paragraphs
        elif level == 2:
            # Keep the first paragraph (usually the summary) and any paragraphs with important keywords
            important_keywords = ['critical', 'important', 'significant', 'key', 'main', 'primary']
            
            filtered_paragraphs = [paragraphs[0]]  # Always keep the first paragraph
            
            for paragraph in paragraphs[1:]:
                if any(re.search(rf'\b{keyword}\b', paragraph, re.IGNORECASE) for keyword in important_keywords):
                    filtered_paragraphs.append(paragraph)
            
            # If we kept too little, add back some paragraphs
            if len(filtered_paragraphs) < max(2, len(paragraphs) // 3):
                filtered_paragraphs.extend(paragraphs[1:max(2, len(paragraphs) // 3)])
                
            return '\n\n'.join(filtered_paragraphs)
            
        # For significant reduction, keep only the summary
        else:
            # Keep only the first paragraph or generate a summary
            return paragraphs[0]
    
    async def _increase_detail_level(self, content: str, level: int = 1) -> str:
        """
        Increase the detail level of the content.
        
        Args:
            content: The content to adjust
            level: The level of increase (1=slight, 2=moderate, 3=significant)
            
        Returns:
            The adjusted content
        """
        # In a real implementation, this would add more details, examples, etc.
        # For now, we'll just return the original content
        return content
    
    async def _adjust_terminology(
        self,
        content: str,
        terminology_level: str
    ) -> str:
        """
        Adjust the terminology used in the content.
        
        Args:
            content: The content to adjust
            terminology_level: The target terminology level
            
        Returns:
            The adjusted content
        """
        if terminology_level == "basic":
            return await self._simplify_terminology(content, 1)
        elif terminology_level == "non_technical":
            return await self._simplify_terminology(content, 2)
        elif terminology_level == "advanced" or terminology_level == "expert":
            return await self._elevate_terminology(content, 1)
        else:
            return content
    
    async def _simplify_terminology(self, content: str, level: int = 1) -> str:
        """
        Simplify the terminology used in the content.
        
        Args:
            content: The content to adjust
            level: The level of simplification (1=slight, 2=significant)
            
        Returns:
            The adjusted content
        """
        # Replace technical terms with simpler alternatives
        simplifications = {
            r'\bimplementation\b': 'creation',
            r'\butilization\b': 'use',
            r'\bfunctionality\b': 'feature',
            r'\barchitecture\b': 'structure',
            r'\binterface\b': 'connection',
            r'\bintegration\b': 'combining',
            r'\boptimization\b': 'improvement',
            r'\bvalidation\b': 'checking',
            r'\bauthentication\b': 'login process',
            r'\bauthorization\b': 'permission checking',
            r'\bconfiguration\b': 'setup',
            r'\binitialization\b': 'startup',
            r'\binstantiation\b': 'creation',
            r'\bpersistence\b': 'saving',
            r'\bserialization\b': 'data conversion',
            r'\bdeserialization\b': 'data reading',
            r'\bsynchronization\b': 'coordination',
            r'\bparameterization\b': 'customization'
        }
        
        # Apply more aggressive simplification for higher levels
        if level >= 2:
            more_simplifications = {
                r'\balgorithm\b': 'step-by-step process',
                r'\bfunction\b': 'task',
                r'\bmethod\b': 'way of doing something',
                r'\bparameter\b': 'setting',
                r'\bvariable\b': 'changeable value',
                r'\bobject\b': 'thing',
                r'\bclass\b': 'type of thing',
                r'\binterface\b': 'way to connect',
                r'\bimplementation\b': 'actual working version',
                r'\binstance\b': 'specific example',
                r'\bcompilation\b': 'preparation',
                r'\bexecution\b': 'running',
                r'\bdebug\b': 'fix problems',
                r'\bstack trace\b': 'error details',
                r'\bexception\b': 'error',
                r'\bmemory leak\b': 'resource waste',
                r'\bgarbage collection\b': 'automatic cleanup',
                r'\basynchronous\b': 'happening in the background',
                r'\bsynchronous\b': 'happening right away',
                r'\bconcurrency\b': 'doing multiple things at once',
                r'\bthreading\b': 'handling multiple tasks',
                r'\brecursion\b': 'repeating process'
            }
            simplifications.update(more_simplifications)
        
        result = content
        for pattern, replacement in simplifications.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    async def _elevate_terminology(self, content: str, level: int = 1) -> str:
        """
        Elevate the terminology used in the content.
        
        Args:
            content: The content to adjust
            level: The level of elevation (1=slight, 2=significant)
            
        Returns:
            The adjusted content
        """
        # Replace simple terms with more technical alternatives
        elevations = {
            r'\buse\b': 'utilization',
            r'\bfeature\b': 'functionality',
            r'\bstructure\b': 'architecture',
            r'\bconnection\b': 'interface',
            r'\bcombining\b': 'integration',
            r'\bimprovement\b': 'optimization',
            r'\bchecking\b': 'validation',
            r'\blogin process\b': 'authentication',
            r'\bpermission checking\b': 'authorization',
            r'\bsetup\b': 'configuration',
            r'\bstartup\b': 'initialization',
            r'\bcreation\b': 'instantiation',
            r'\bsaving\b': 'persistence',
            r'\bdata conversion\b': 'serialization',
            r'\bdata reading\b': 'deserialization',
            r'\bcoordination\b': 'synchronization',
            r'\bcustomization\b': 'parameterization'
        }
        
        result = content
        for pattern, replacement in elevations.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    async def _adjust_focus(
        self,
        content: str,
        focus_areas: List[str]
    ) -> str:
        """
        Adjust the focus of the content to emphasize certain areas.
        
        Args:
            content: The content to adjust
            focus_areas: The areas to focus on
            
        Returns:
            The adjusted content
        """
        return await self._shift_focus(content, focus_areas)
    
    async def _shift_focus(
        self,
        content: str,
        focus_areas: List[str]
    ) -> str:
        """
        Shift the focus of the content to emphasize certain areas.
        
        Args:
            content: The content to adjust
            focus_areas: The areas to focus on
            
        Returns:
            The adjusted content
        """
        # In a real implementation, this would reorganize and emphasize
        # content related to the specified focus areas
        
        # For now, we'll just add a focus note at the beginning
        if "business_impact" in focus_areas:
            business_note = "\n\n**Business Impact Focus**: This analysis emphasizes business impact and ROI considerations.\n\n"
            return business_note + content
        elif "educational" in focus_areas:
            educational_note = "\n\n**Learning Focus**: This explanation includes educational content to help build understanding.\n\n"
            return educational_note + content
        elif "security_vulnerabilities" in focus_areas:
            security_note = "\n\n**Security Focus**: This analysis emphasizes security implications and vulnerability remediation.\n\n"
            return security_note + content
        elif "architecture" in focus_areas:
            architecture_note = "\n\n**Architecture Focus**: This analysis emphasizes architectural implications and design considerations.\n\n"
            return architecture_note + content
        
        return content
    
    async def _reorganize_content(
        self,
        content: str,
        audience: Audience
    ) -> str:
        """
        Reorganize the content based on audience preferences.
        
        Args:
            content: The content to reorganize
            audience: The target audience
            
        Returns:
            The reorganized content
        """
        # In a real implementation, this would reorganize the content
        # based on audience preferences
        return content
    
    async def _apply_personalization(
        self,
        content: str,
        personalization_context: PersonalizationContext
    ) -> str:
        """
        Apply personalization to the content based on user context.
        
        Args:
            content: The content to personalize
            personalization_context: The personalization context
            
        Returns:
            The personalized content
        """
        # In a real implementation, this would apply personalization
        # based on the user's preferences, history, etc.
        
        # For now, we'll just add a personalized greeting
        experience_level = personalization_context.experience_level.value.capitalize()
        role = personalization_context.role.value.replace('_', ' ').capitalize()
        
        personalized_greeting = f"\n\n*Personalized for {experience_level} {role}*\n\n"
        
        # Add references to known technologies if relevant
        known_techs = personalization_context.known_technologies
        if known_techs:
            tech_references = []
            for tech in known_techs:
                if tech.lower() in content.lower():
                    tech_references.append(tech)
            
            if tech_references:
                personalized_greeting += f"*Referencing your experience with: {', '.join(tech_references)}*\n\n"
        
        return personalized_greeting + content
