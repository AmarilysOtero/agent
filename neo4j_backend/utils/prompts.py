"""Prompts for entity and relationship extraction using LLMs"""

ENTITY_EXTRACTION_PROMPT = """You are extracting named entities from a document chunk.

Extract entities of the following types: {entity_types}

For each entity, provide:
1. name: The entity name/mention as it appears in the text
2. type: One of the specified entity types
3. confidence: A score from 0.0 to 1.0 indicating extraction confidence
4. context: A brief snippet from the text showing the entity in context (max 100 chars)

TEXT TO ANALYZE:
{chunk_text}

Return your response as a JSON object with this structure:
{{
    "entities": [
        {{
            "name": "entity name",
            "type": "entity type",
            "confidence": 0.95,
            "context": "text snippet with entity..."
        }}
    ]
}}

Guidelines:
- Only extract entities that clearly belong to one of the specified types
- Use exact names as they appear in the text
- For People, include full names when available
- For Organizations, use official names
- For Locations, be specific (city, country, region)
- For Concepts, identify key ideas, technologies, methodologies
- For Events, identify specific happenings or occurrences
- For Products, identify specific products, tools, or services
- Confidence should reflect how certain you are about the entity type and extraction
- Include entities only if confidence >= 0.7
"""

RELATIONSHIP_EXTRACTION_PROMPT = """You are extracting typed relationships between entities in a document chunk.

The following entities have been identified:
{entity_names}

Extract relationships between these entities from the following text:

TEXT TO ANALYZE:
{chunk_text}

For each relationship, provide:
1. subject: The name of the subject entity (must be from the entity list)
2. relationship_type: The type of relationship (see list below)
3. object: The name of the object entity (must be from the entity list)
4. confidence: A score from 0.0 to 1.0 indicating extraction confidence

Relationship types to use:
- WORKS_FOR: Person works for Organization
- LOCATED_IN: Entity is located in Location
- PART_OF: Entity is part of another Entity
- COLLABORATES_WITH: Entity collaborates with another Entity
- CREATES: Entity creates another Entity (e.g., Person creates Product)
- MENTIONS: Entity mentions another Entity
- RELATED_TO: Generic relationship when more specific type doesn't apply
- CAUSES: Entity causes another Entity or Event
- PARTICIPATES_IN: Entity participates in Event
- OWNS: Entity owns another Entity

Return your response as a JSON object with this structure:
{{
    "relationships": [
        {{
            "subject": "entity1 name",
            "relationship_type": "WORKS_FOR",
            "object": "entity2 name",
            "confidence": 0.90
        }}
    ]
}}

Guidelines:
- Only extract relationships explicitly stated or strongly implied in the text
- Both subject and object must be from the provided entity list
- Use the most specific relationship type that applies
- Confidence should reflect how certain you are about the relationship
- Include relationships only if confidence >= 0.7
- A relationship can be directional (subject -> object) or bidirectional
"""

ENTITY_CANONICALIZATION_PROMPT = """You are determining if two entity names refer to the same real-world entity.

Entity 1:
Name: {name1}
Type: {type1}
Context: {context1}

Entity 2:
Name: {name2}
Type: {type2}
Context: {context2}

Determine if these are the same entity. Consider:
- Name variations (abbreviations, nicknames, full vs shortened names)
- Context clues
- Entity types (must match)

Return your response as a JSON object:
{{
    "is_same_entity": true/false,
    "confidence": 0.95,
    "reasoning": "Brief explanation of why they are/aren't the same"
}}

Guidelines:
- Different entity types -> definitely not the same (confidence 1.0)
- Exact name match + same type -> likely same (confidence 0.9+)
- Similar names + same type + similar context -> possibly same (check carefully)
- Only mark as same entity if confidence >= 0.85
"""
