"""Intelligent memory analyzer that understands context and significance."""

import re
from dataclasses import dataclass
from enum import Enum

from src.memory import MemoryType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SignificanceLevel(Enum):
    """Levels of significance for memory formation."""

    CRITICAL = 1.0  # Must remember (names, personal info, preferences)
    HIGH = 0.8  # Very important (goals, decisions, plans)
    MEDIUM = 0.6  # Worth remembering (facts, experiences)
    LOW = 0.4  # Casual (general conversation)
    MINIMAL = 0.2  # Background noise


@dataclass
class MemorySignal:
    """Represents a detected memory-worthy signal in content."""

    content: str
    significance: float
    memory_type: MemoryType
    context: str
    entities: list[str]
    reasoning: str


class IntelligentMemoryAnalyzer:
    """AI-powered memory analyzer that understands what should be remembered."""

    def __init__(self):
        """Initialize the intelligent analyzer."""
        self._initialize_patterns()
        self._initialize_context_analyzers()

    def _initialize_patterns(self):
        """Initialize semantic patterns for different types of memorable content."""
        # Identity and personal information patterns
        self.identity_patterns = [
            r"(?:my name is|i'?m|call me|i am called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:you can call me|people call me|known as)\s+([A-Z][a-z]+)",
            r"i'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"my\s+(?:first\s+)?name\s+is\s+([A-Z][a-z]+)",
        ]

        # Contact and location patterns
        self.contact_patterns = [
            r"(?:my\s+)?(?:phone\s+)?(?:number\s+is|phone\s+is|cell\s+is)\s*([\d\-\(\)\s\.]+)",
            r"(?:my\s+)?email\s+(?:is\s+|address\s+is\s+)?([\w\.\-]+@[\w\.\-]+\.\w+)",
            r"i\s+live\s+(?:in\s+|at\s+)(.+?)(?:\.|,|$)",
            r"my\s+address\s+is\s+(.+?)(?:\.|,|$)",
        ]

        # Preference and personality patterns
        self.preference_patterns = [
            r"i\s+(?:really\s+)?(?:love|like|enjoy|prefer|hate|dislike|can'?t\s+stand)\s+(.+?)(?:\.|,|because|$)",
            r"my\s+favorite\s+(.+?)\s+is\s+(.+?)(?:\.|,|$)",
            r"i\s+(?:always|never|usually|typically|often)\s+(.+?)(?:\.|,|$)",
            r"i'?m\s+(?:allergic\s+to|afraid\s+of|passionate\s+about)\s+(.+?)(?:\.|,|$)",
        ]

        # Goal and plan patterns
        self.goal_patterns = [
            r"(?:my\s+goal\s+is|i\s+want\s+to|i\s+plan\s+to|i'?m\s+trying\s+to|i\s+hope\s+to)\s+(.+?)(?:\.|,|$)",
            r"(?:i\s+need\s+to|i\s+should|i\s+must)\s+(?:remember\s+to\s+)?(.+?)(?:\.|,|$)",
            r"(?:remind\s+me\s+to|don'?t\s+forget\s+to|make\s+sure\s+to)\s+(.+?)(?:\.|,|$)",
        ]

        # Experience and event patterns
        self.experience_patterns = [
            r"(?:yesterday|today|tomorrow|last\s+\w+|next\s+\w+|this\s+\w+)\s+(.+?)(?:\.|,|$)",
            r"(?:i\s+went\s+to|i\s+visited|i\s+attended|i\s+met)\s+(.+?)(?:\.|,|$)",
            r"(?:something\s+(?:amazing|terrible|important|interesting)\s+happened|i\s+had\s+a\s+great\s+time)\s*(.+?)(?:\.|,|$)",
        ]

        # Knowledge and learning patterns
        self.knowledge_patterns = [
            r"(?:did\s+you\s+know|i\s+learned|i\s+discovered|interesting\s+fact)\s+(.+?)(?:\.|,|$)",
            r"(?:the\s+key\s+is|what\s+matters\s+is|important\s+thing\s+is)\s+(.+?)(?:\.|,|$)",
            r"(?:always\s+remember|never\s+forget)\s+(?:that\s+)?(.+?)(?:\.|,|$)",
        ]

    def _initialize_context_analyzers(self):
        """Initialize contextual analysis functions."""
        self.context_analyzers = [
            self._analyze_emotional_context,
            self._analyze_temporal_context,
            self._analyze_relationship_context,
            self._analyze_decision_context,
            self._analyze_instruction_context,
        ]

    def analyze_memory_significance(
        self, content: str, conversation_context: list[str] = None
    ) -> list[MemorySignal]:
        """
        Analyze content to determine what should be remembered and why.

        Args:
            content: The text content to analyze
            conversation_context: Recent conversation messages for context

        Returns:
            List of memory signals with significance levels
        """
        signals = []
        content_lower = content.lower().strip()

        # Quick early returns for performance
        if not content_lower or len(content_lower) < 5:
            return signals

        # Skip analysis for very common/short phrases that are unlikely to be memorable
        common_phrases = {"ok", "yes", "no", "sure", "thanks", "thank you", "hello", "hi", "bye"}
        if content_lower in common_phrases:
            return signals

        # Skip if content is mostly punctuation or numbers
        if len([c for c in content_lower if c.isalpha()]) < len(content_lower) * 0.6:
            return signals

        # Extract different types of memorable information (most important first for early wins)
        signals.extend(self._extract_identity_info(content))
        if len(signals) >= 3:  # Early return if we found significant identity info
            return self._finalize_signals(signals, content, conversation_context)

        signals.extend(self._extract_contact_info(content))
        signals.extend(self._extract_preferences(content))
        signals.extend(self._extract_goals_and_plans(content))
        signals.extend(self._extract_experiences(content))
        signals.extend(self._extract_knowledge(content))

        return self._finalize_signals(signals, content, conversation_context)

    def _finalize_signals(
        self, signals: list, content: str, conversation_context: list[str]
    ) -> list:
        """Finalize signals with contextual analysis and scoring."""
        # Apply contextual analysis only if we have some initial signals
        if signals:
            for analyzer in self.context_analyzers:
                contextual_signals = analyzer(content, conversation_context or [])
                signals.extend(contextual_signals)

            # Analyze overall conversation flow and significance
            if conversation_context:
                signals.extend(
                    self._analyze_conversation_significance(content, conversation_context)
                )

        # Remove duplicates and merge related signals
        signals = self._deduplicate_and_merge_signals(signals)

        # Apply final significance scoring
        signals = self._apply_intelligent_scoring(signals, content, conversation_context)

        return signals

    def _extract_identity_info(self, content: str) -> list[MemorySignal]:
        """Extract identity and personal information."""
        signals = []

        for pattern in self.identity_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if self._is_valid_name(name):
                    signals.append(
                        MemorySignal(
                            content=f"User's name is {name}",
                            significance=SignificanceLevel.CRITICAL.value,
                            memory_type=MemoryType.LONG_TERM,
                            context="personal_identity",
                            entities=[name],
                            reasoning=f"Extracted personal name from pattern: {pattern}",
                        )
                    )

        return signals

    def _extract_contact_info(self, content: str) -> list[MemorySignal]:
        """Extract contact information."""
        signals = []

        for pattern in self.contact_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                contact_info = match.group(1).strip()
                if self._is_valid_contact(contact_info, pattern):
                    contact_type = self._determine_contact_type(pattern)
                    signals.append(
                        MemorySignal(
                            content=f"User's {contact_type}: {contact_info}",
                            significance=SignificanceLevel.CRITICAL.value,
                            memory_type=MemoryType.LONG_TERM,
                            context="contact_information",
                            entities=[contact_info],
                            reasoning=f"Extracted {contact_type} information",
                        )
                    )

        return signals

    def _extract_preferences(self, content: str) -> list[MemorySignal]:
        """Extract preferences and personality traits."""
        signals = []

        for pattern in self.preference_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                preference = match.group(1).strip()
                if len(preference) > 2 and not self._is_common_word(preference):
                    sentiment = self._determine_sentiment(content, match.start())
                    signals.append(
                        MemorySignal(
                            content=f"User {sentiment} {preference}",
                            significance=SignificanceLevel.HIGH.value,
                            memory_type=MemoryType.LONG_TERM,
                            context="personal_preferences",
                            entities=[preference],
                            reasoning=f"Extracted preference with {sentiment} sentiment",
                        )
                    )

        return signals

    def _extract_goals_and_plans(self, content: str) -> list[MemorySignal]:
        """Extract goals, plans, and intentions."""
        signals = []

        for pattern in self.goal_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                goal = match.group(1).strip()
                if len(goal) > 3:
                    signals.append(
                        MemorySignal(
                            content=f"User goal/plan: {goal}",
                            significance=SignificanceLevel.HIGH.value,
                            memory_type=MemoryType.LONG_TERM,
                            context="goals_and_plans",
                            entities=[goal],
                            reasoning="Extracted goal or plan statement",
                        )
                    )

        return signals

    def _extract_experiences(self, content: str) -> list[MemorySignal]:
        """Extract experiences and events."""
        signals = []

        for pattern in self.experience_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                experience = match.group(1).strip() if match.lastindex else match.group(0)
                if len(experience) > 5:
                    signals.append(
                        MemorySignal(
                            content=f"User experience: {experience}",
                            significance=SignificanceLevel.MEDIUM.value,
                            memory_type=MemoryType.EPISODIC,
                            context="personal_experiences",
                            entities=[experience],
                            reasoning="Extracted temporal experience or event",
                        )
                    )

        return signals

    def _extract_knowledge(self, content: str) -> list[MemorySignal]:
        """Extract factual knowledge and insights."""
        signals = []

        for pattern in self.knowledge_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                knowledge = match.group(1).strip()
                if len(knowledge) > 5:
                    signals.append(
                        MemorySignal(
                            content=f"Knowledge: {knowledge}",
                            significance=SignificanceLevel.MEDIUM.value,
                            memory_type=MemoryType.SEMANTIC,
                            context="factual_knowledge",
                            entities=[knowledge],
                            reasoning="Extracted factual knowledge or insight",
                        )
                    )

        return signals

    def _analyze_emotional_context(self, content: str, context: list[str]) -> list[MemorySignal]:
        """Analyze emotional context and significance."""
        emotional_indicators = [
            ("love", "hate", "excited", "thrilled", "devastated", "heartbroken"),
            ("important", "crucial", "critical", "essential", "vital"),
            ("amazing", "incredible", "wonderful", "terrible", "awful"),
            ("proud", "ashamed", "grateful", "regret", "sorry"),
        ]

        signals = []
        content_lower = content.lower()

        for indicators in emotional_indicators:
            for indicator in indicators:
                if indicator in content_lower:
                    # Extract the sentence containing the emotional indicator
                    sentences = re.split(r"[.!?]+", content)
                    for sentence in sentences:
                        if indicator in sentence.lower():
                            signals.append(
                                MemorySignal(
                                    content=sentence.strip(),
                                    significance=SignificanceLevel.HIGH.value,
                                    memory_type=MemoryType.EPISODIC,
                                    context="emotional_significance",
                                    entities=[indicator],
                                    reasoning=f"High emotional content detected: {indicator}",
                                )
                            )
                            break

        return signals

    def _analyze_temporal_context(self, content: str, context: list[str]) -> list[MemorySignal]:
        """Analyze temporal significance."""
        time_indicators = [
            r"\b(?:tomorrow|next\s+\w+|in\s+\w+\s+days?|this\s+\w+)\b",
            r"\b(?:deadline|due|appointment|meeting|schedule)\b",
            r"\b(?:birthday|anniversary|graduation|wedding)\b",
        ]

        signals = []

        for pattern in time_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                signals.append(
                    MemorySignal(
                        content=content,
                        significance=SignificanceLevel.HIGH.value,
                        memory_type=MemoryType.EPISODIC,
                        context="temporal_significance",
                        entities=[],
                        reasoning="Contains time-sensitive information",
                    )
                )
                break

        return signals

    def _analyze_relationship_context(self, content: str, context: list[str]) -> list[MemorySignal]:
        """Analyze relationship and social context."""
        relationship_patterns = [
            r"\b(?:my\s+(?:mom|dad|mother|father|sister|brother|wife|husband|partner|friend|boss|colleague))\b",
            r"\b(?:met|introduced|befriended|dating|married|divorced)\b",
            r"\b(?:family|relatives|coworkers|teammates)\b",
        ]

        signals = []

        for pattern in relationship_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                signals.append(
                    MemorySignal(
                        content=content,
                        significance=SignificanceLevel.HIGH.value,
                        memory_type=MemoryType.LONG_TERM,
                        context="relationship_information",
                        entities=[],
                        reasoning="Contains relationship information",
                    )
                )
                break

        return signals

    def _analyze_decision_context(self, content: str, context: list[str]) -> list[MemorySignal]:
        """Analyze decision-making context."""
        decision_patterns = [
            r"\b(?:decided|chose|picked|selected|determined)\b",
            r"\b(?:will|going\s+to|planning\s+to|intend\s+to)\b",
            r"\b(?:never\s+again|always\s+do|always\s+remember)\b",
        ]

        signals = []

        for pattern in decision_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                signals.append(
                    MemorySignal(
                        content=content,
                        significance=SignificanceLevel.MEDIUM.value,
                        memory_type=MemoryType.LONG_TERM,
                        context="decision_making",
                        entities=[],
                        reasoning="Contains decision or commitment",
                    )
                )
                break

        return signals

    def _analyze_instruction_context(self, content: str, context: list[str]) -> list[MemorySignal]:
        """Analyze instructional or procedural context."""
        instruction_patterns = [
            r"\b(?:remember\s+to|don'?t\s+forget|make\s+sure|always|never)\b",
            r"\b(?:password|pin|code|setting|configuration)\b",
            r"\b(?:how\s+to|steps|process|procedure|method)\b",
        ]

        signals = []

        for pattern in instruction_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                signals.append(
                    MemorySignal(
                        content=content,
                        significance=SignificanceLevel.HIGH.value,
                        memory_type=MemoryType.SEMANTIC,
                        context="instructional_content",
                        entities=[],
                        reasoning="Contains instructions or procedures",
                    )
                )
                break

        return signals

    def _analyze_conversation_significance(
        self, content: str, context: list[str]
    ) -> list[MemorySignal]:
        """Analyze overall conversation flow for significance."""
        signals = []

        # Check if this is a continuation of an important topic
        important_topics = ["name", "personal", "important", "remember", "family", "work", "goal"]

        context_text = " ".join(context[-3:]).lower()  # Last 3 messages
        content.lower()

        topic_matches = sum(1 for topic in important_topics if topic in context_text)

        if topic_matches >= 2:
            signals.append(
                MemorySignal(
                    content=content,
                    significance=SignificanceLevel.MEDIUM.value,
                    memory_type=MemoryType.SHORT_TERM,
                    context="conversation_continuity",
                    entities=[],
                    reasoning="Part of ongoing important conversation thread",
                )
            )

        return signals

    def _deduplicate_and_merge_signals(self, signals: list[MemorySignal]) -> list[MemorySignal]:
        """Remove duplicates and merge related signals."""
        if not signals:
            return signals

        # Group by content similarity
        unique_signals = []
        for signal in signals:
            is_duplicate = False
            for existing in unique_signals:
                if self._signals_are_similar(signal, existing):
                    # Merge into existing signal with higher significance
                    if signal.significance > existing.significance:
                        existing.significance = signal.significance
                        existing.reasoning += f" | {signal.reasoning}"
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_signals.append(signal)

        return unique_signals

    def _apply_intelligent_scoring(
        self, signals: list[MemorySignal], content: str, context: list[str]
    ) -> list[MemorySignal]:
        """Apply intelligent scoring based on multiple factors."""
        for signal in signals:
            # Boost significance based on multiple factors
            boost = 0.0

            # Length and detail boost
            if len(signal.content) > 50:
                boost += 0.1

            # Personal pronoun boost
            if any(pronoun in signal.content.lower() for pronoun in ["i", "my", "me", "myself"]):
                boost += 0.15

            # Emotional intensity boost
            emotion_words = [
                "love",
                "hate",
                "excited",
                "important",
                "critical",
                "amazing",
                "terrible",
            ]
            emotion_count = sum(1 for word in emotion_words if word in signal.content.lower())
            boost += emotion_count * 0.1

            # Context relevance boost
            if context and len(context) > 1:
                recent_context = " ".join(context[-2:]).lower()
                shared_words = set(signal.content.lower().split()) & set(recent_context.split())
                if len(shared_words) > 2:
                    boost += 0.1

            # Apply boost but cap at 1.0
            signal.significance = min(1.0, signal.significance + boost)

        return signals

    # Helper methods
    def _is_valid_name(self, name: str) -> bool:
        """Check if extracted text is a valid name."""
        if len(name) < 2 or len(name) > 50:
            return False
        # Check if it's likely a name (starts with capital, only letters and spaces)
        return bool(re.match(r"^[A-Z][a-zA-Z\s]+$", name))

    def _is_valid_contact(self, contact: str, pattern: str) -> bool:
        """Check if extracted text is valid contact info."""
        if "email" in pattern:
            return "@" in contact and "." in contact
        elif "phone" in pattern:
            return len(re.sub(r"[^\d]", "", contact)) >= 10
        return len(contact) > 3

    def _determine_contact_type(self, pattern: str) -> str:
        """Determine type of contact info from pattern."""
        if "email" in pattern:
            return "email"
        elif "phone" in pattern:
            return "phone number"
        elif "address" in pattern:
            return "address"
        return "contact info"

    def _is_common_word(self, word: str) -> bool:
        """Check if word is too common to be memorable."""
        common_words = {
            "it",
            "that",
            "this",
            "what",
            "when",
            "where",
            "how",
            "why",
            "the",
            "a",
            "an",
        }
        return word.lower().strip() in common_words

    def _determine_sentiment(self, content: str, position: int) -> str:
        """Determine sentiment around a specific position in text."""
        positive_words = ["love", "like", "enjoy", "prefer", "favorite"]
        negative_words = ["hate", "dislike", "can't stand", "avoid"]

        window = content[max(0, position - 20) : position + 20].lower()

        if any(word in window for word in positive_words):
            return "likes"
        elif any(word in window for word in negative_words):
            return "dislikes"
        return "mentioned"

    def _signals_are_similar(self, signal1: MemorySignal, signal2: MemorySignal) -> bool:
        """Check if two signals are similar enough to merge."""
        # Simple similarity check - can be enhanced with embedding similarity
        words1 = set(signal1.content.lower().split())
        words2 = set(signal2.content.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union > 0.6  # 60% word overlap
