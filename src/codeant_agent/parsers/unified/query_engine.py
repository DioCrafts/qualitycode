"""
Engine de Queries Unificado Cross-Language.

Este módulo implementa un sistema de consultas unificado que permite realizar
búsquedas y análisis sobre múltiples lenguajes de programación usando una
sintaxis de consulta coherente y expresiva.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from .unified_ast import (
    UnifiedAST,
    UnifiedNode,
    UnifiedNodeType,
    SemanticNodeType,
    NodeId,
)

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Tipos de consultas soportadas."""
    NODE_SEARCH = "node_search"
    PATTERN_MATCH = "pattern_match"
    SEMANTIC_SEARCH = "semantic_search"
    STRUCTURAL_SEARCH = "structural_search"
    CROSS_LANGUAGE_COMPARISON = "cross_language_comparison"
    METRICS_QUERY = "metrics_query"


class QueryOperator(str, Enum):
    """Operadores de consulta."""
    EQUALS = "="
    NOT_EQUALS = "!="
    CONTAINS = "CONTAINS"
    STARTS_WITH = "STARTS_WITH"
    ENDS_WITH = "ENDS_WITH"
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    IN = "IN"
    NOT_IN = "NOT_IN"
    EXISTS = "EXISTS"
    NOT_EXISTS = "NOT_EXISTS"


@dataclass
class QueryFilter:
    """Filtro de consulta."""
    field: str
    operator: QueryOperator
    value: Any
    negated: bool = False


@dataclass
class QueryProjection:
    """Proyección de consulta."""
    field: str
    alias: Optional[str] = None
    transform: Optional[str] = None


@dataclass
class QueryAggregation:
    """Agregación de consulta."""
    function: str  # COUNT, SUM, AVG, MIN, MAX, etc.
    field: str
    alias: Optional[str] = None
    group_by: Optional[List[str]] = None


@dataclass
class UnifiedQuery:
    """Consulta unificada."""
    query_string: str
    query_type: QueryType
    target_languages: List[str] = field(default_factory=list)
    filters: List[QueryFilter] = field(default_factory=list)
    projections: List[QueryProjection] = field(default_factory=list)
    aggregations: List[QueryAggregation] = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: List[str] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """Resultado de una consulta."""
    query: UnifiedQuery
    results: List[Dict[str, Any]]
    total_count: int
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossLanguageQueryResult:
    """Resultado de consulta cross-language."""
    query: UnifiedQuery
    results_by_language: Dict[str, List[Dict[str, Any]]]
    cross_language_patterns: List[Dict[str, Any]]
    summary: Dict[str, Any]
    execution_time_ms: float


class QueryError(Exception):
    """Error en la ejecución de consultas."""
    
    def __init__(self, message: str, query: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.query = query
        self.details = details or {}


class QueryParser:
    """Parser de consultas unificadas."""
    
    def __init__(self):
        self.operators = {
            "=": QueryOperator.EQUALS,
            "!=": QueryOperator.NOT_EQUALS,
            "CONTAINS": QueryOperator.CONTAINS,
            "STARTS_WITH": QueryOperator.STARTS_WITH,
            "ENDS_WITH": QueryOperator.ENDS_WITH,
            ">": QueryOperator.GREATER_THAN,
            "<": QueryOperator.LESS_THAN,
            ">=": QueryOperator.GREATER_EQUAL,
            "<=": QueryOperator.LESS_EQUAL,
            "IN": QueryOperator.IN,
            "NOT_IN": QueryOperator.NOT_IN,
            "EXISTS": QueryOperator.EXISTS,
            "NOT_EXISTS": QueryOperator.NOT_EXISTS,
        }
    
    def parse(self, query_string: str) -> UnifiedQuery:
        """Parsea una cadena de consulta en un objeto UnifiedQuery."""
        try:
            # Parsear el tipo de consulta
            query_type = self._parse_query_type(query_string)
            
            # Parsear filtros
            filters = self._parse_filters(query_string)
            
            # Parsear proyecciones
            projections = self._parse_projections(query_string)
            
            # Parsear agregaciones
            aggregations = self._parse_aggregations(query_string)
            
            # Parsear lenguajes objetivo
            target_languages = self._parse_target_languages(query_string)
            
            # Parsear límites y ordenamiento
            limit, offset = self._parse_limit_offset(query_string)
            order_by = self._parse_order_by(query_string)
            group_by = self._parse_group_by(query_string)
            
            return UnifiedQuery(
                query_string=query_string,
                query_type=query_type,
                target_languages=target_languages,
                filters=filters,
                projections=projections,
                aggregations=aggregations,
                limit=limit,
                offset=offset,
                order_by=order_by,
                group_by=group_by,
            )
            
        except Exception as e:
            raise QueryError(f"Error parsing query: {str(e)}", query=query_string)
    
    def _parse_query_type(self, query_string: str) -> QueryType:
        """Parsea el tipo de consulta."""
        query_lower = query_string.upper()
        
        if "FIND" in query_lower:
            return QueryType.NODE_SEARCH
        elif "MATCH" in query_lower:
            return QueryType.PATTERN_MATCH
        elif "SEMANTIC" in query_lower:
            return QueryType.SEMANTIC_SEARCH
        elif "STRUCTURAL" in query_lower:
            return QueryType.STRUCTURAL_SEARCH
        elif "CROSS" in query_lower and "LANGUAGE" in query_lower:
            return QueryType.CROSS_LANGUAGE_COMPARISON
        elif "METRICS" in query_lower:
            return QueryType.METRICS_QUERY
        else:
            return QueryType.NODE_SEARCH  # Default
    
    def _parse_filters(self, query_string: str) -> List[QueryFilter]:
        """Parsea los filtros de la consulta."""
        filters = []
        
        # Buscar patrones WHERE field operator value
        where_pattern = r'WHERE\s+(\w+)\s+([^\s]+)\s+([^ANDOR]+)'
        matches = re.finditer(where_pattern, query_string, re.IGNORECASE)
        
        for match in matches:
            field = match.group(1)
            operator_str = match.group(2)
            value_str = match.group(3).strip()
            
            operator = self.operators.get(operator_str, QueryOperator.EQUALS)
            value = self._parse_value(value_str)
            
            filters.append(QueryFilter(
                field=field,
                operator=operator,
                value=value
            ))
        
        return filters
    
    def _parse_projections(self, query_string: str) -> List[QueryProjection]:
        """Parsea las proyecciones de la consulta."""
        projections = []
        
        # Buscar patrones SELECT field1, field2, ...
        select_pattern = r'SELECT\s+([^FROM]+)'
        match = re.search(select_pattern, query_string, re.IGNORECASE)
        
        if match:
            fields_str = match.group(1)
            fields = [f.strip() for f in fields_str.split(',')]
            
            for field in fields:
                if ' AS ' in field.upper():
                    field_parts = field.split(' AS ')
                    projections.append(QueryProjection(
                        field=field_parts[0].strip(),
                        alias=field_parts[1].strip()
                    ))
                else:
                    projections.append(QueryProjection(field=field))
        
        return projections
    
    def _parse_aggregations(self, query_string: str) -> List[QueryAggregation]:
        """Parsea las agregaciones de la consulta."""
        aggregations = []
        
        # Buscar patrones COUNT(field), SUM(field), etc.
        agg_pattern = r'(\w+)\((\w+)\)'
        matches = re.finditer(agg_pattern, query_string, re.IGNORECASE)
        
        for match in matches:
            function = match.group(1).upper()
            field = match.group(2)
            
            aggregations.append(QueryAggregation(
                function=function,
                field=field
            ))
        
        return aggregations
    
    def _parse_target_languages(self, query_string: str) -> List[str]:
        """Parsea los lenguajes objetivo."""
        languages = []
        
        # Buscar patrones IN [language1, language2, ...]
        lang_pattern = r'IN\s*\[([^\]]+)\]'
        match = re.search(lang_pattern, query_string, re.IGNORECASE)
        
        if match:
            langs_str = match.group(1)
            languages = [lang.strip() for lang in langs_str.split(',')]
        
        return languages
    
    def _parse_limit_offset(self, query_string: str) -> tuple[Optional[int], Optional[int]]:
        """Parsea límites y offset."""
        limit = None
        offset = None
        
        # Buscar LIMIT number
        limit_match = re.search(r'LIMIT\s+(\d+)', query_string, re.IGNORECASE)
        if limit_match:
            limit = int(limit_match.group(1))
        
        # Buscar OFFSET number
        offset_match = re.search(r'OFFSET\s+(\d+)', query_string, re.IGNORECASE)
        if offset_match:
            offset = int(offset_match.group(1))
        
        return limit, offset
    
    def _parse_order_by(self, query_string: str) -> List[str]:
        """Parsea el ordenamiento."""
        order_by = []
        
        # Buscar ORDER BY field1, field2, ...
        order_pattern = r'ORDER BY\s+([^;]+)'
        match = re.search(order_pattern, query_string, re.IGNORECASE)
        
        if match:
            fields_str = match.group(1)
            order_by = [f.strip() for f in fields_str.split(',')]
        
        return order_by
    
    def _parse_group_by(self, query_string: str) -> List[str]:
        """Parsea el agrupamiento."""
        group_by = []
        
        # Buscar GROUP BY field1, field2, ...
        group_pattern = r'GROUP BY\s+([^;]+)'
        match = re.search(group_pattern, query_string, re.IGNORECASE)
        
        if match:
            fields_str = match.group(1)
            group_by = [f.strip() for f in fields_str.split(',')]
        
        return group_by
    
    def _parse_value(self, value_str: str) -> Any:
        """Parsea un valor de filtro."""
        value_str = value_str.strip()
        
        # String literal
        if value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]
        
        # Number
        if value_str.replace('.', '').replace('-', '').isdigit():
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        
        # Boolean
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        # List
        if value_str.startswith('[') and value_str.endswith(']'):
            items = value_str[1:-1].split(',')
            return [self._parse_value(item.strip()) for item in items]
        
        # Default to string
        return value_str


class QueryExecutor:
    """Ejecutor de consultas."""
    
    def __init__(self):
        self.node_visitors = {
            QueryType.NODE_SEARCH: self._execute_node_search,
            QueryType.PATTERN_MATCH: self._execute_pattern_match,
            QueryType.SEMANTIC_SEARCH: self._execute_semantic_search,
            QueryType.STRUCTURAL_SEARCH: self._execute_structural_search,
            QueryType.CROSS_LANGUAGE_COMPARISON: self._execute_cross_language_comparison,
            QueryType.METRICS_QUERY: self._execute_metrics_query,
        }
    
    async def execute(self, query: UnifiedQuery, asts: List[UnifiedAST]) -> QueryResult:
        """Ejecuta una consulta sobre una lista de ASTs."""
        start_time = datetime.now()
        
        try:
            # Filtrar ASTs por lenguaje si se especifica
            filtered_asts = self._filter_asts_by_language(asts, query.target_languages)
            
            # Ejecutar la consulta según su tipo
            executor = self.node_visitors.get(query.query_type)
            if not executor:
                raise QueryError(f"Unsupported query type: {query.query_type}")
            
            results = await executor(query, filtered_asts)
            
            # Aplicar filtros
            filtered_results = self._apply_filters(results, query.filters)
            
            # Aplicar proyecciones
            projected_results = self._apply_projections(filtered_results, query.projections)
            
            # Aplicar agregaciones
            if query.aggregations:
                projected_results = self._apply_aggregations(projected_results, query.aggregations, query.group_by)
            
            # Aplicar ordenamiento
            if query.order_by:
                projected_results = self._apply_ordering(projected_results, query.order_by)
            
            # Aplicar límites
            if query.limit:
                projected_results = projected_results[:query.limit]
            
            # Calcular tiempo de ejecución
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                query=query,
                results=projected_results,
                total_count=len(projected_results),
                execution_time_ms=execution_time,
                metadata={
                    "asts_processed": len(filtered_asts),
                    "languages": list(set(ast.language for ast in filtered_asts)),
                }
            )
            
        except Exception as e:
            if isinstance(e, QueryError):
                raise
            raise QueryError(f"Error executing query: {str(e)}", query=query.query_string)
    
    def _filter_asts_by_language(self, asts: List[UnifiedAST], target_languages: List[str]) -> List[UnifiedAST]:
        """Filtra ASTs por lenguaje objetivo."""
        if not target_languages:
            return asts
        
        return [ast for ast in asts if ast.language in target_languages]
    
    async def _execute_node_search(self, query: UnifiedQuery, asts: List[UnifiedAST]) -> List[Dict[str, Any]]:
        """Ejecuta búsqueda de nodos."""
        results = []
        
        for ast in asts:
            # Buscar nodos que coincidan con los criterios
            matching_nodes = self._find_matching_nodes(ast, query)
            
            for node in matching_nodes:
                result = {
                    "node_id": str(node.id),
                    "node_type": node.node_type.value,
                    "semantic_type": node.semantic_type.value,
                    "name": node.name,
                    "language": ast.language,
                    "file_path": str(ast.file_path),
                    "position": {
                        "start_line": node.position.start_line if node.position else None,
                        "start_column": node.position.start_column if node.position else None,
                        "end_line": node.position.end_line if node.position else None,
                        "end_column": node.position.end_column if node.position else None,
                    } if node.position else None,
                }
                results.append(result)
        
        return results
    
    async def _execute_pattern_match(self, query: UnifiedQuery, asts: List[UnifiedAST]) -> List[Dict[str, Any]]:
        """Ejecuta búsqueda de patrones."""
        results = []
        
        for ast in asts:
            # Buscar patrones en el AST
            patterns = self._find_patterns(ast, query)
            results.extend(patterns)
        
        return results
    
    async def _execute_semantic_search(self, query: UnifiedQuery, asts: List[UnifiedAST]) -> List[Dict[str, Any]]:
        """Ejecuta búsqueda semántica."""
        results = []
        
        for ast in asts:
            # Buscar elementos semánticos
            semantic_elements = self._find_semantic_elements(ast, query)
            results.extend(semantic_elements)
        
        return results
    
    async def _execute_structural_search(self, query: UnifiedQuery, asts: List[UnifiedAST]) -> List[Dict[str, Any]]:
        """Ejecuta búsqueda estructural."""
        results = []
        
        for ast in asts:
            # Buscar estructuras
            structures = self._find_structures(ast, query)
            results.extend(structures)
        
        return results
    
    async def _execute_cross_language_comparison(self, query: UnifiedQuery, asts: List[UnifiedAST]) -> List[Dict[str, Any]]:
        """Ejecuta comparación cross-language."""
        results = []
        
        # Comparar ASTs entre sí
        for i in range(len(asts)):
            for j in range(i + 1, len(asts)):
                comparison = self._compare_asts(asts[i], asts[j])
                results.append(comparison)
        
        return results
    
    async def _execute_metrics_query(self, query: UnifiedQuery, asts: List[UnifiedAST]) -> List[Dict[str, Any]]:
        """Ejecuta consulta de métricas."""
        results = []
        
        for ast in asts:
            metrics = {
                "language": ast.language,
                "file_path": str(ast.file_path),
                "node_count": ast.get_node_count(),
                "depth": ast.get_depth(),
                "complexity_score": ast.get_complexity_score(),
                "function_count": len(ast.find_nodes_by_type(UnifiedNodeType.FUNCTION_DECLARATION)),
                "class_count": len(ast.find_nodes_by_type(UnifiedNodeType.CLASS_DECLARATION)),
                "import_count": len(ast.semantic_info.imports),
                "symbol_count": len(ast.semantic_info.symbols),
            }
            results.append(metrics)
        
        return results
    
    def _find_matching_nodes(self, ast: UnifiedAST, query: UnifiedQuery) -> List[UnifiedNode]:
        """Encuentra nodos que coincidan con los criterios de búsqueda."""
        matching_nodes = []
        
        def traverse(node: UnifiedNode):
            if self._node_matches_criteria(node, query):
                matching_nodes.append(node)
            
            for child in node.children:
                traverse(child)
        
        traverse(ast.root_node)
        return matching_nodes
    
    def _node_matches_criteria(self, node: UnifiedNode, query: UnifiedQuery) -> bool:
        """Determina si un nodo coincide con los criterios de búsqueda."""
        # Implementación básica - se expandirá
        return True
    
    def _find_patterns(self, ast: UnifiedAST, query: UnifiedQuery) -> List[Dict[str, Any]]:
        """Encuentra patrones en un AST."""
        # Implementación básica
        return []
    
    def _find_semantic_elements(self, ast: UnifiedAST, query: UnifiedQuery) -> List[Dict[str, Any]]:
        """Encuentra elementos semánticos en un AST."""
        # Implementación básica
        return []
    
    def _find_structures(self, ast: UnifiedAST, query: UnifiedQuery) -> List[Dict[str, Any]]:
        """Encuentra estructuras en un AST."""
        # Implementación básica
        return []
    
    def _compare_asts(self, ast1: UnifiedAST, ast2: UnifiedAST) -> Dict[str, Any]:
        """Compara dos ASTs."""
        return {
            "language1": ast1.language,
            "language2": ast2.language,
            "similarity_score": 0.5,  # Placeholder
            "common_patterns": [],
            "differences": [],
        }
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: List[QueryFilter]) -> List[Dict[str, Any]]:
        """Aplica filtros a los resultados."""
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            if all(self._apply_filter(result, filter) for filter in filters):
                filtered_results.append(result)
        
        return filtered_results
    
    def _apply_filter(self, result: Dict[str, Any], filter: QueryFilter) -> bool:
        """Aplica un filtro individual a un resultado."""
        if filter.field not in result:
            return False
        
        value = result[filter.field]
        filter_value = filter.value
        
        if filter.operator == QueryOperator.EQUALS:
            return value == filter_value
        elif filter.operator == QueryOperator.NOT_EQUALS:
            return value != filter_value
        elif filter.operator == QueryOperator.CONTAINS:
            return str(filter_value) in str(value)
        elif filter.operator == QueryOperator.STARTS_WITH:
            return str(value).startswith(str(filter_value))
        elif filter.operator == QueryOperator.ENDS_WITH:
            return str(value).endswith(str(filter_value))
        elif filter.operator == QueryOperator.GREATER_THAN:
            return value > filter_value
        elif filter.operator == QueryOperator.LESS_THAN:
            return value < filter_value
        elif filter.operator == QueryOperator.GREATER_EQUAL:
            return value >= filter_value
        elif filter.operator == QueryOperator.LESS_EQUAL:
            return value <= filter_value
        elif filter.operator == QueryOperator.IN:
            return value in filter_value
        elif filter.operator == QueryOperator.NOT_IN:
            return value not in filter_value
        elif filter.operator == QueryOperator.EXISTS:
            return value is not None
        elif filter.operator == QueryOperator.NOT_EXISTS:
            return value is None
        
        return True
    
    def _apply_projections(self, results: List[Dict[str, Any]], projections: List[QueryProjection]) -> List[Dict[str, Any]]:
        """Aplica proyecciones a los resultados."""
        if not projections:
            return results
        
        projected_results = []
        
        for result in results:
            projected_result = {}
            
            for projection in projections:
                if projection.field in result:
                    value = result[projection.field]
                    
                    # Aplicar transformación si se especifica
                    if projection.transform:
                        value = self._apply_transform(value, projection.transform)
                    
                    # Usar alias si se especifica
                    field_name = projection.alias or projection.field
                    projected_result[field_name] = value
            
            projected_results.append(projected_result)
        
        return projected_results
    
    def _apply_transform(self, value: Any, transform: str) -> Any:
        """Aplica una transformación a un valor."""
        # Implementación básica - se expandirá
        return value
    
    def _apply_aggregations(self, results: List[Dict[str, Any]], aggregations: List[QueryAggregation], group_by: List[str]) -> List[Dict[str, Any]]:
        """Aplica agregaciones a los resultados."""
        if not aggregations:
            return results
        
        # Implementación básica - se expandirá
        return results
    
    def _apply_ordering(self, results: List[Dict[str, Any]], order_by: List[str]) -> List[Dict[str, Any]]:
        """Aplica ordenamiento a los resultados."""
        if not order_by:
            return results
        
        def sort_key(result):
            key_values = []
            for field in order_by:
                if field.startswith('-'):
                    # Orden descendente
                    field_name = field[1:]
                    value = result.get(field_name, 0)
                    key_values.append(-value if isinstance(value, (int, float)) else value)
                else:
                    # Orden ascendente
                    value = result.get(field, 0)
                    key_values.append(value)
            return key_values
        
        return sorted(results, key=sort_key)


class ResultAggregator:
    """Agregador de resultados cross-language."""
    
    async def aggregate_cross_language(self, language_results: Dict[str, QueryResult]) -> CrossLanguageQueryResult:
        """Agrega resultados de múltiples lenguajes."""
        start_time = datetime.now()
        
        # Combinar resultados de todos los lenguajes
        all_results = []
        for language, result in language_results.items():
            for item in result.results:
                item['language'] = language
                all_results.append(item)
        
        # Encontrar patrones cross-language
        cross_language_patterns = self._find_cross_language_patterns(all_results)
        
        # Generar resumen
        summary = self._generate_cross_language_summary(language_results, all_results)
        
        # Calcular tiempo de ejecución
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        return CrossLanguageQueryResult(
            query=next(iter(language_results.values())).query,
            results_by_language={lang: result.results for lang, result in language_results.items()},
            cross_language_patterns=cross_language_patterns,
            summary=summary,
            execution_time_ms=execution_time,
        )
    
    def _find_cross_language_patterns(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encuentra patrones que aparecen en múltiples lenguajes."""
        patterns = []
        
        # Agrupar por tipo de nodo
        node_types = {}
        for result in all_results:
            node_type = result.get('node_type')
            if node_type:
                if node_type not in node_types:
                    node_types[node_type] = []
                node_types[node_type].append(result)
        
        # Identificar patrones cross-language
        for node_type, results in node_types.items():
            languages = list(set(result.get('language') for result in results))
            if len(languages) > 1:
                patterns.append({
                    'pattern_type': node_type,
                    'languages': languages,
                    'occurrences': len(results),
                    'description': f"Pattern '{node_type}' found in {len(languages)} languages",
                })
        
        return patterns
    
    def _generate_cross_language_summary(self, language_results: Dict[str, QueryResult], all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera un resumen del análisis cross-language."""
        return {
            'total_results': len(all_results),
            'languages_analyzed': list(language_results.keys()),
            'results_per_language': {lang: len(result.results) for lang, result in language_results.items()},
            'execution_times': {lang: result.execution_time_ms for lang, result in language_results.items()},
            'most_common_patterns': self._get_most_common_patterns(all_results),
        }
    
    def _get_most_common_patterns(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Obtiene los patrones más comunes."""
        pattern_counts = {}
        for result in all_results:
            pattern = result.get('node_type', 'unknown')
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return [
            {'pattern': pattern, 'count': count}
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]


class UnifiedQueryEngine:
    """Motor principal de consultas unificadas."""
    
    def __init__(self):
        self.query_parser = QueryParser()
        self.query_executor = QueryExecutor()
        self.result_aggregator = ResultAggregator()
        self.query_cache = {}
    
    async def execute_query(self, query: str, asts: List[UnifiedAST]) -> QueryResult:
        """Ejecuta una consulta sobre una lista de ASTs."""
        # Parsear la consulta
        parsed_query = self.query_parser.parse(query)
        
        # Verificar cache
        cache_key = self._generate_cache_key(parsed_query, asts)
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            if not self._is_cache_expired(cached_result):
                return cached_result['result']
        
        # Ejecutar la consulta
        result = await self.query_executor.execute(parsed_query, asts)
        
        # Cachear el resultado
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now(),
            'ttl': 300,  # 5 minutos
        }
        
        return result
    
    async def execute_cross_language_query(self, query: str, asts: List[UnifiedAST]) -> CrossLanguageQueryResult:
        """Ejecuta una consulta cross-language."""
        parsed_query = self.query_parser.parse(query)
        
        # Agrupar ASTs por lenguaje
        language_groups = {}
        for ast in asts:
            if ast.language not in language_groups:
                language_groups[ast.language] = []
            language_groups[ast.language].append(ast)
        
        # Ejecutar consulta en cada grupo de lenguaje
        language_results = {}
        for language, ast_group in language_groups.items():
            result = await self.query_executor.execute(parsed_query, ast_group)
            language_results[language] = result
        
        # Agregar resultados cross-language
        aggregated_result = await self.result_aggregator.aggregate_cross_language(language_results)
        
        return aggregated_result
    
    def _generate_cache_key(self, query: UnifiedQuery, asts: List[UnifiedAST]) -> str:
        """Genera una clave de cache para una consulta."""
        # Implementación básica - se expandirá
        return f"{query.query_string}_{len(asts)}"
    
    def _is_cache_expired(self, cached_item: Dict[str, Any]) -> bool:
        """Determina si un elemento del cache ha expirado."""
        timestamp = cached_item['timestamp']
        ttl = cached_item['ttl']
        now = datetime.now()
        
        return (now - timestamp).total_seconds() > ttl
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del motor de consultas."""
        return {
            'cache_size': len(self.query_cache),
            'cache_hit_rate': 0.0,  # Se calcularía con métricas reales
            'supported_query_types': list(QueryType),
            'supported_operators': list(QueryOperator),
        }
