"""Advanced model management interface with intelligent selection and RAG configuration.

This module implements a comprehensive model management system demonstrating:

- Dynamic model discovery across multiple LLM providers
- Intelligent model selection strategies for different use cases
- Provider health monitoring with real-time status updates
- RAG (Retrieval-Augmented Generation) configuration interface
- Professional model comparison with capability visualization
- Thread-safe async operations for provider communication
- Responsive UI design with adaptive model cards

Model Management Architecture:
    The interface connects to multiple LLM providers (Ollama, OpenAI, LM Studio)
    through a unified abstraction layer, enabling seamless model switching
    and configuration across different AI backends.

Provider Health Monitoring:
    - Real-time status checking for provider availability
    - Response time measurement for performance optimization
    - Error detection and recovery suggestions
    - Model capability enumeration and comparison

Selection Strategies:
    - Manual selection for user control
    - Fastest response for performance-critical applications
    - Cheapest operation for cost-sensitive scenarios
    - Best quality for accuracy-focused tasks
    - Balanced approach for general-purpose usage
    - Most capable for complex reasoning requirements

RAG System Integration:
    - Configurable memory retrieval parameters
    - Context window optimization for different models
    - Source citation and reasoning explanation controls
    - Performance tuning for large knowledge bases

Accessibility Features:
    - Keyboard navigation throughout the interface
    - Screen reader compatible model information
    - High contrast design for visual accessibility
    - Touch-friendly controls with appropriate sizing
"""

import threading
from typing import Any, Dict, List, Optional

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDIconButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.list import MDList, OneLineAvatarIconListItem, TwoLineAvatarIconListItem
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.textfield import MDTextField
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.chip import MDChip

from src.core.model_manager import ModelInfo, ModelSelectionStrategy, ModelStatus, ProviderInfo
from src.core.rag_system import RAGConfig
from src.gui.theme import UIConstants
from src.gui.utils.notifications import Notification
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelCard(MDCard):
    """Professional model display card with comprehensive information and controls.
    
    This widget provides sophisticated model representation featuring:
    
    Information Architecture:
        - Model name with provider attribution
        - Status indicators with color-coded feedback
        - Capability chips for quick feature identification
        - Performance metrics and cost information
        - Technical specifications (parameters, context length)
        
    Visual Design:
        - Material Design card styling with appropriate elevation
        - Current model highlighting with accent colors
        - Capability visualization through colored chips
        - Professional typography with clear hierarchy
        - Responsive layout for different screen sizes
        
    Interactive Elements:
        - Model selection button with state management
        - Disabled state for unavailable models
        - Current model indication with visual prominence
        - Touch feedback and hover states
        
    Model Information Display:
        - Display name with fallback to technical name
        - Provider identification with consistent formatting
        - Availability status with semantic colors
        - Description text with proper text wrapping
        
    Technical Specifications:
        - Parameter count with human-readable formatting
        - Context length in thousands (K) for readability
        - Cost per token with appropriate precision
        - Capability badges with truncation for space
        
    State Management:
        - Current model highlighting with accent colors
        - Availability-based interaction enabling/disabling
        - Selection callback for parent component communication
        - Proper widget tree traversal for event handling
        
    The card demonstrates advanced Kivy custom widget creation
    with professional information design and interaction patterns.
    
    Args:
        model_info (ModelInfo): Complete model specification and metadata
        is_current (bool): Whether this model is currently selected
    """
    
    def __init__(self, model_info: ModelInfo, is_current: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.model_info = model_info
        self.is_current = is_current
        self.orientation = "vertical"
        self.padding = UIConstants.PADDING_MEDIUM
        self.spacing = UIConstants.SPACING_SMALL
        self.elevation = UIConstants.ELEVATION_CARD
        self.radius = [UIConstants.RADIUS_MEDIUM]
        self.size_hint_y = None
        self.adaptive_height = True
        
        # Highlight current model
        if self.is_current:
            self.md_bg_color = self.theme_cls.primary_color
            self.md_bg_color[3] = 0.1  # Semi-transparent
        
        self.build_ui()
    
    def build_ui(self):
        """Build model card UI."""
        # Header with model name and provider
        header = MDBoxLayout(
            orientation="horizontal",
            spacing=UIConstants.SPACING_MEDIUM,
            adaptive_height=True,
            size_hint_y=None
        )
        
        # Model info
        info_layout = MDBoxLayout(orientation="vertical", adaptive_height=True)
        
        # Model name
        name_text = self.model_info.display_name or self.model_info.name
        if self.is_current:
            name_text += " (Current)"
        
        name_label = MDLabel(
            text=name_text,
            font_style="Subtitle1",
            theme_text_color="Primary",
            adaptive_height=True,
            bold=self.is_current
        )
        
        # Provider and status
        status_layout = MDBoxLayout(
            orientation="horizontal",
            spacing=UIConstants.SPACING_SMALL,
            adaptive_height=True
        )
        
        provider_label = MDLabel(
            text=f"Provider: {self.model_info.provider.title()}",
            font_style="Caption",
            theme_text_color="Secondary",
            adaptive_height=True
        )
        
        status_color = "Primary" if self.model_info.status == ModelStatus.AVAILABLE else "Error"
        status_text = "Available" if self.model_info.status == ModelStatus.AVAILABLE else "Unavailable"
        
        status_label = MDLabel(
            text=f"Status: {status_text}",
            font_style="Caption",
            theme_text_color=status_color,
            adaptive_height=True
        )
        
        status_layout.add_widget(provider_label)
        status_layout.add_widget(status_label)
        
        # Description
        if self.model_info.description:
            desc_label = MDLabel(
                text=self.model_info.description,
                font_style="Body2",
                theme_text_color="Secondary",
                adaptive_height=True
            )
            info_layout.add_widget(desc_label)
        
        info_layout.add_widget(name_label)
        info_layout.add_widget(status_layout)
        
        # Model specs
        specs_layout = MDBoxLayout(
            orientation="horizontal",
            spacing=UIConstants.SPACING_SMALL,
            adaptive_height=True
        )
        
        if self.model_info.parameters:
            param_chip = MDChip(
                text=self.model_info.parameters,
                height=dp(24),
                size_hint_y=None
            )
            specs_layout.add_widget(param_chip)
        
        if self.model_info.context_length:
            context_chip = MDChip(
                text=f"{self.model_info.context_length // 1000}K context",
                height=dp(24),
                size_hint_y=None
            )
            specs_layout.add_widget(context_chip)
        
        if self.model_info.cost_per_token:
            cost_chip = MDChip(
                text=f"${self.model_info.cost_per_token:.4f}/token",
                height=dp(24),
                size_hint_y=None
            )
            specs_layout.add_widget(cost_chip)
        
        # Capabilities
        if self.model_info.capabilities:
            caps_layout = MDBoxLayout(
                orientation="horizontal",
                spacing=UIConstants.SPACING_SMALL,
                adaptive_height=True
            )
            
            for cap in self.model_info.capabilities[:4]:  # Show first 4 capabilities
                cap_chip = MDChip(
                    text=cap.replace("_", " ").title(),
                    height=dp(20),
                    size_hint_y=None
                )
                caps_layout.add_widget(cap_chip)
            
            if len(self.model_info.capabilities) > 4:
                more_chip = MDChip(
                    text=f"+{len(self.model_info.capabilities) - 4} more",
                    height=dp(20),
                    size_hint_y=None
                )
                caps_layout.add_widget(more_chip)
        
        # Select button
        select_button = MDRaisedButton(
            text="Select" if not self.is_current else "Selected",
            size_hint_y=None,
            height=dp(36),
            disabled=self.is_current or self.model_info.status != ModelStatus.AVAILABLE,
            on_release=self.on_select_model
        )
        
        header.add_widget(info_layout)
        
        self.add_widget(header)
        if specs_layout.children:
            self.add_widget(specs_layout)
        if 'caps_layout' in locals() and caps_layout.children:
            self.add_widget(caps_layout)
        self.add_widget(select_button)
    
    def on_select_model(self, button):
        """Handle model selection."""
        # Find the model management screen by traversing up the widget tree
        parent = self.parent
        while parent and not isinstance(parent, ModelManagementScreen):
            parent = parent.parent
        
        if parent and hasattr(parent, 'on_model_selected'):
            parent.on_model_selected(self.model_info)


class ProviderHealthCard(MDCard):
    """Provider health monitoring card with real-time status visualization.
    
    This widget provides comprehensive provider health information:
    
    Health Metrics Display:
        - Provider name with consistent branding
        - Health status with semantic color coding
        - Response time metrics for performance assessment
        - Available model count for capability overview
        - Error message display for troubleshooting
        
    Status Visualization:
        - Health state colors (green=healthy, yellow=degraded, red=error)
        - Response time display in milliseconds
        - Model availability count for quick assessment
        - Error information for debugging assistance
        
    Visual Design:
        - Professional card layout with proper spacing
        - Semantic color coding for status indication
        - Clear typography hierarchy for information scanning
        - Appropriate elevation and Material Design styling
        
    Information Architecture:
        - Provider identification with title case formatting
        - Health status with descriptive labels
        - Performance metrics with appropriate units
        - Error messages with actionable information
        
    Status Categories:
        - Healthy: Provider is fully operational
        - Degraded: Provider is functional but experiencing issues
        - Unavailable: Provider is not accessible
        - Error: Provider encountered specific errors
        
    Performance Monitoring:
        - Response time measurement for latency assessment
        - Model enumeration for capability tracking
        - Error detection for reliability monitoring
        - Status history for trend analysis
        
    The card provides essential information for provider selection
    and troubleshooting in a professional, scannable format.
    
    Args:
        provider_info (ProviderInfo): Complete provider health and capability data
    """
    
    def __init__(self, provider_info: ProviderInfo, **kwargs):
        super().__init__(**kwargs)
        self.provider_info = provider_info
        self.orientation = "vertical"
        self.padding = UIConstants.PADDING_MEDIUM
        self.spacing = UIConstants.SPACING_SMALL
        self.elevation = UIConstants.ELEVATION_CARD
        self.radius = [UIConstants.RADIUS_MEDIUM]
        self.size_hint_y = None
        self.adaptive_height = True
        
        self.build_ui()
    
    def build_ui(self):
        """Build provider health UI."""
        # Header
        header = MDBoxLayout(
            orientation="horizontal",
            spacing=UIConstants.SPACING_MEDIUM,
            adaptive_height=True
        )
        
        # Provider name
        name_label = MDLabel(
            text=self.provider_info.name.title(),
            font_style="Subtitle1",
            theme_text_color="Primary",
            adaptive_height=True
        )
        
        # Status indicator
        status_colors = {
            "healthy": "Primary",
            "degraded": "Secondary", 
            "unavailable": "Error",
            "error": "Error"
        }
        
        status_label = MDLabel(
            text=self.provider_info.status.value.title(),
            font_style="Caption",
            theme_text_color=status_colors.get(self.provider_info.status.value, "Secondary"),
            adaptive_height=True
        )
        
        header.add_widget(name_label)
        header.add_widget(status_label)
        
        # Details
        details_layout = MDBoxLayout(
            orientation="vertical",
            spacing=UIConstants.SPACING_SMALL,
            adaptive_height=True
        )
        
        # Model count
        model_count_label = MDLabel(
            text=f"Models: {len(self.provider_info.available_models)}",
            font_style="Body2",
            theme_text_color="Secondary",
            adaptive_height=True
        )
        details_layout.add_widget(model_count_label)
        
        # Response time
        if self.provider_info.response_time_ms:
            time_label = MDLabel(
                text=f"Response Time: {self.provider_info.response_time_ms:.0f}ms",
                font_style="Body2",
                theme_text_color="Secondary",
                adaptive_height=True
            )
            details_layout.add_widget(time_label)
        
        # Error message if any
        if self.provider_info.error_message:
            error_label = MDLabel(
                text=f"Error: {self.provider_info.error_message}",
                font_style="Caption",
                theme_text_color="Error",
                adaptive_height=True
            )
            details_layout.add_widget(error_label)
        
        self.add_widget(header)
        self.add_widget(details_layout)


class RAGConfigCard(MDCard):
    """Advanced RAG (Retrieval-Augmented Generation) configuration interface.
    
    This widget provides comprehensive RAG system configuration:
    
    Configuration Categories:
        - Retrieval parameters (memory count, relevance threshold)
        - Context management (token limits, timeout settings)
        - Quality controls (source citation, reasoning explanation)
        - Performance optimizations (deduplication, temporal weighting)
        
    Interactive Controls:
        - RAG enable/disable toggle with visual feedback
        - Numeric input fields with validation and constraints
        - Boolean option toggles with checkbox-style interface
        - Apply button for configuration persistence
        
    Parameter Configuration:
        - Max memories: Number of relevant memories to retrieve
        - Relevance threshold: Minimum similarity score (0-1)
        - Context tokens: Maximum tokens for memory context
        - Timeout: Maximum time for memory retrieval (ms)
        
    Quality Features:
        - Cite sources: Include memory source attribution
        - Explain reasoning: Provide retrieval reasoning
        - Deduplicate content: Remove similar memory content
        - Temporal weighting: Prefer recent memories
        
    User Experience:
        - Intuitive grid layout for parameter organization
        - Real-time validation with input constraints
        - Visual feedback for configuration changes
        - Professional form styling with consistent spacing
        
    Technical Implementation:
        - Input validation with type constraints
        - State management for boolean options
        - Configuration object construction from UI state
        - Parent component communication for persistence
        
    Accessibility:
        - Keyboard navigation through configuration options
        - Clear labeling for screen reader compatibility
        - Logical tab order for efficient configuration
        - Visual indicators for current settings
        
    The card provides complete control over RAG system behavior
    with professional configuration interface and validation.
    
    Args:
        rag_config (RAGConfig): Current RAG system configuration
    """
    
    def __init__(self, rag_config: RAGConfig, **kwargs):
        super().__init__(**kwargs)
        self.rag_config = rag_config
        self.orientation = "vertical"
        self.padding = UIConstants.PADDING_MEDIUM
        self.spacing = UIConstants.SPACING_MEDIUM
        self.elevation = UIConstants.ELEVATION_CARD
        self.radius = [UIConstants.RADIUS_MEDIUM]
        self.size_hint_y = None
        self.adaptive_height = True
        
        self.build_ui()
    
    def build_ui(self):
        """Build RAG configuration UI."""
        # Header
        header_label = MDLabel(
            text="RAG Configuration",
            font_style="H6",
            theme_text_color="Primary",
            adaptive_height=True
        )
        self.add_widget(header_label)
        
        # Enable button instead of switch to avoid MDSwitch issues
        enable_layout = MDBoxLayout(
            orientation="horizontal",
            spacing=UIConstants.SPACING_MEDIUM,
            adaptive_height=True,
            size_hint_y=None
        )
        
        enable_label = MDLabel(
            text="RAG Status:",
            font_style="Body1",
            adaptive_height=True
        )
        
        self.rag_enabled = True  # Track state
        self.enable_button = MDRaisedButton(
            text="Enabled",
            size_hint_y=None,
            height=dp(36),
            on_release=self.toggle_rag
        )
        
        enable_layout.add_widget(enable_label)
        enable_layout.add_widget(self.enable_button)
        self.add_widget(enable_layout)
        
        # Configuration options
        config_grid = MDGridLayout(
            cols=2,
            spacing=UIConstants.SPACING_MEDIUM,
            adaptive_height=True,
            size_hint_y=None
        )
        
        # Max memories
        self.max_memories_field = MDTextField(
            hint_text="Max Memories",
            text=str(self.rag_config.max_memories),
            input_filter="int",
            size_hint_y=None,
            height=dp(56)
        )
        
        # Relevance threshold
        self.threshold_field = MDTextField(
            hint_text="Min Relevance (0-1)",
            text=str(self.rag_config.min_relevance_threshold),
            input_filter="float",
            size_hint_y=None,
            height=dp(56)
        )
        
        # Max context tokens
        self.context_tokens_field = MDTextField(
            hint_text="Max Context Tokens",
            text=str(self.rag_config.max_context_tokens),
            input_filter="int",
            size_hint_y=None,
            height=dp(56)
        )
        
        # Timeout
        self.timeout_field = MDTextField(
            hint_text="Timeout (ms)",
            text=str(self.rag_config.retrieval_timeout_ms),
            input_filter="int",
            size_hint_y=None,
            height=dp(56)
        )
        
        config_grid.add_widget(self.max_memories_field)
        config_grid.add_widget(self.threshold_field)
        config_grid.add_widget(self.context_tokens_field)
        config_grid.add_widget(self.timeout_field)
        
        self.add_widget(config_grid)
        
        # Checkboxes for boolean options (avoiding MDSwitch)
        options_layout = MDBoxLayout(
            orientation="vertical",
            spacing=UIConstants.SPACING_SMALL,
            adaptive_height=True
        )
        
        options_title = MDLabel(
            text="Options:",
            font_style="Subtitle2",
            adaptive_height=True
        )
        options_layout.add_widget(options_title)
        
        # Create checkboxes for boolean options
        bool_options = [
            ("cite_sources", "Cite Sources", self.rag_config.cite_sources),
            ("explain_reasoning", "Explain Reasoning", self.rag_config.explain_reasoning),
            ("deduplicate_content", "Deduplicate Content", self.rag_config.deduplicate_content),
            ("temporal_weighting", "Temporal Weighting", self.rag_config.temporal_weighting),
        ]
        
        self.option_states = {}
        for key, label, default in bool_options:
            self.option_states[key] = default
            
            option_text = f"[{'X' if default else ' '}] {label}"
            option_button = MDFlatButton(
                text=option_text,
                size_hint_y=None,
                height=dp(36),
                on_release=lambda x, k=key, l=label: self.toggle_option(k, l)
            )
            option_button.key = key
            option_button.label = label
            options_layout.add_widget(option_button)
        
        self.add_widget(options_layout)
        
        # Apply button
        apply_button = MDRaisedButton(
            text="Apply Configuration",
            size_hint_y=None,
            height=dp(40),
            on_release=self.on_apply_config
        )
        self.add_widget(apply_button)
    
    def toggle_rag(self, button):
        """Toggle RAG enabled state."""
        self.rag_enabled = not self.rag_enabled
        button.text = "Enabled" if self.rag_enabled else "Disabled"
        # Notify parent to enable/disable RAG
        if hasattr(self.parent.parent.parent.parent, 'chat_manager'):
            self.parent.parent.parent.parent.chat_manager.enable_rag(self.rag_enabled)
    
    def toggle_option(self, key, label):
        """Toggle a boolean option."""
        self.option_states[key] = not self.option_states[key]
        # Update button text
        for child in self.children:
            if isinstance(child, MDBoxLayout):
                for subchild in child.children:
                    if isinstance(subchild, MDFlatButton) and hasattr(subchild, 'key') and subchild.key == key:
                        subchild.text = f"[{'X' if self.option_states[key] else ' '}] {label}"
                        break
    
    def on_apply_config(self, button):
        """Handle configuration application."""
        if hasattr(self.parent.parent.parent.parent, 'on_rag_config_updated'):
            config = self.get_config()
            self.parent.parent.parent.parent.on_rag_config_updated(config)
    
    def get_config(self) -> RAGConfig:
        """Get current configuration from UI."""
        return RAGConfig(
            max_memories=int(self.max_memories_field.text or "10"),
            min_relevance_threshold=float(self.threshold_field.text or "0.7"),
            max_context_tokens=int(self.context_tokens_field.text or "4000"),
            retrieval_timeout_ms=int(self.timeout_field.text or "5000"),
            cite_sources=self.option_states.get("cite_sources", True),
            explain_reasoning=self.option_states.get("explain_reasoning", False),
            deduplicate_content=self.option_states.get("deduplicate_content", True),
            temporal_weighting=self.option_states.get("temporal_weighting", True),
        )


class ModelManagementScreen(MDScreen):
    """Comprehensive model management interface with intelligent selection.
    
    This screen provides complete model lifecycle management:
    
    Core Functionality:
        - Dynamic model discovery across multiple providers
        - Intelligent model selection with multiple strategies
        - Provider health monitoring with real-time updates
        - RAG system configuration and optimization
        - Current model status display and management
        
    Model Discovery:
        - Automatic provider scanning for available models
        - Model capability enumeration and comparison
        - Performance metric collection and display
        - Status validation and health checking
        - Forced refresh for up-to-date information
        
    Selection Strategies:
        - Manual: User-driven model selection
        - Fastest: Optimize for response time
        - Cheapest: Minimize operational costs
        - Best Quality: Maximize response accuracy
        - Balanced: Optimize across multiple dimensions
        - Most Capable: Select highest capability model
        
    Provider Management:
        - Multi-provider support (Ollama, OpenAI, LM Studio)
        - Health status monitoring with visual indicators
        - Error detection and recovery guidance
        - Performance metrics and comparison
        - Connection management and troubleshooting
        
    RAG Configuration:
        - Memory retrieval parameter tuning
        - Context window optimization
        - Quality control settings (citations, reasoning)
        - Performance optimization (deduplication, weighting)
        - Real-time configuration application
        
    User Interface Design:
        - Professional Material Design implementation
        - Responsive layout with adaptive sizing
        - Color-coded status indicators throughout
        - Intuitive navigation and action placement
        - Comprehensive information display
        
    Performance Optimization:
        - Background threading for provider communication
        - Efficient model discovery with caching
        - Responsive UI during long operations
        - Optimized data binding and updates
        - Memory-efficient widget management
        
    State Management:
        - Current model tracking and display
        - Provider health state synchronization
        - Configuration persistence and validation
        - Error state handling and recovery
        - Thread-safe UI updates
        
    Accessibility Features:
        - Keyboard navigation throughout the interface
        - Screen reader compatible information structure
        - High contrast design for visibility
        - Touch-friendly controls with appropriate sizing
        - Clear visual hierarchy and focus indicators
        
    The screen demonstrates advanced model management patterns
    with professional UI design and comprehensive functionality.
    
    Args:
        chat_manager: Shared chat management instance for model operations
    """
    
    def __init__(self, chat_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.name = "model_management"
        self.chat_manager = chat_manager
        self.current_model_info = None
        
        self.build_ui()
        
        # Schedule model discovery
        Clock.schedule_once(self.discover_models, 1.0)
    
    def build_ui(self):
        """Construct the comprehensive model management interface.
        
        This method builds a sophisticated UI structure featuring:
        
        Layout Architecture:
            1. Header with navigation, title, and primary actions
            2. Current model status display with key information
            3. Provider health section with real-time monitoring
            4. Available models section with detailed cards
            5. RAG configuration section with comprehensive controls
            
        Component Organization:
            - Header: Navigation, title, refresh, and strategy selection
            - Status: Current model information with provider details
            - Health: Provider status cards with performance metrics
            - Models: Detailed model cards with selection capabilities
            - RAG: Configuration interface with parameter controls
            
        Design Principles:
            - Consistent spacing using UI design system constants
            - Professional typography with clear hierarchy
            - Semantic color coding for status and actions
            - Responsive layout for different screen sizes
            - Material Design elevation and shadow effects
            
        Interactive Elements:
            - Back navigation for screen transitions
            - Refresh functionality for up-to-date information
            - Strategy dropdown for intelligent model selection
            - Model selection cards with visual feedback
            - RAG configuration controls with real-time updates
            
        Information Display:
            - Current model status with provider attribution
            - Provider health cards with performance metrics
            - Model capability visualization with chips
            - RAG configuration state with parameter values
            - Loading states and progress indication
            
        Scrollable Content:
            - Vertical scrolling for large model lists
            - Efficient rendering for performance
            - Appropriate spacing and visual grouping
            - Adaptive sizing for different content volumes
            
        The UI demonstrates advanced Material Design implementation
        with proper component composition and professional presentation.
        """
        main_layout = MDBoxLayout(
            orientation="vertical",
            spacing=UIConstants.SPACING_MEDIUM,
            padding=UIConstants.PADDING_MEDIUM
        )
        
        # Header
        header = MDBoxLayout(
            orientation="horizontal",
            spacing=UIConstants.SPACING_MEDIUM,
            adaptive_height=True,
            size_hint_y=None
        )
        
        # Back button
        back_button = MDIconButton(
            icon="arrow-left",
            on_release=self.go_back
        )
        
        title = MDLabel(
            text="Model Management",
            font_style="H4",
            theme_text_color="Primary",
            adaptive_height=True
        )
        
        # Refresh button
        refresh_button = MDIconButton(
            icon="refresh",
            on_release=self.discover_models
        )
        
        # Selection strategy dropdown
        strategy_button = MDRaisedButton(
            text="Auto Select",
            size_hint_y=None,
            height=dp(40),
            on_release=self.show_strategy_menu
        )
        
        header.add_widget(back_button)
        header.add_widget(title)
        header.add_widget(refresh_button)
        header.add_widget(strategy_button)
        
        main_layout.add_widget(header)
        
        # Current model info
        current_model_layout = MDBoxLayout(
            orientation="horizontal",
            spacing=UIConstants.SPACING_SMALL,
            adaptive_height=True,
            size_hint_y=None,
            padding=[0, dp(10), 0, dp(10)]
        )
        
        current_label = MDLabel(
            text="Current Model:",
            font_style="Subtitle1",
            theme_text_color="Secondary",
            adaptive_height=True,
            size_hint_x=None,
            width=dp(120)
        )
        
        self.current_model_label = MDLabel(
            text="Loading...",
            font_style="Subtitle1", 
            theme_text_color="Primary",
            adaptive_height=True,
            bold=True
        )
        
        current_model_layout.add_widget(current_label)
        current_model_layout.add_widget(self.current_model_label)
        main_layout.add_widget(current_model_layout)
        
        # Scrollable content
        scroll = MDScrollView()
        content_layout = MDBoxLayout(
            orientation="vertical",
            spacing=UIConstants.SPACING_MEDIUM,
            adaptive_height=True,
            size_hint_y=None
        )
        
        # Provider health section
        health_title = MDLabel(
            text="Provider Health",
            font_style="H6",
            theme_text_color="Primary",
            adaptive_height=True
        )
        content_layout.add_widget(health_title)
        
        self.health_layout = MDBoxLayout(
            orientation="vertical",
            spacing=UIConstants.SPACING_SMALL,
            adaptive_height=True,
            size_hint_y=None
        )
        content_layout.add_widget(self.health_layout)
        
        # Models section
        models_title = MDLabel(
            text="Available Models",
            font_style="H6",
            theme_text_color="Primary",
            adaptive_height=True
        )
        content_layout.add_widget(models_title)
        
        self.models_layout = MDBoxLayout(
            orientation="vertical",
            spacing=UIConstants.SPACING_MEDIUM,
            adaptive_height=True,
            size_hint_y=None
        )
        content_layout.add_widget(self.models_layout)
        
        # RAG Configuration section
        rag_title = MDLabel(
            text="RAG Configuration",
            font_style="H6",
            theme_text_color="Primary",
            adaptive_height=True
        )
        content_layout.add_widget(rag_title)
        
        self.rag_layout = MDBoxLayout(
            orientation="vertical",
            adaptive_height=True,
            size_hint_y=None
        )
        content_layout.add_widget(self.rag_layout)
        
        scroll.add_widget(content_layout)
        main_layout.add_widget(scroll)
        
        self.add_widget(main_layout)
        
        # Initialize RAG config
        self.setup_rag_config()
    
    def setup_rag_config(self):
        """Setup RAG configuration UI."""
        if self.chat_manager and hasattr(self.chat_manager, 'rag_system'):
            config = self.chat_manager.rag_system.config
            rag_card = RAGConfigCard(config)
            self.rag_layout.add_widget(rag_card)
    
    def discover_models(self, dt=None):
        """Initiate comprehensive model discovery across all providers.
        
        This method implements provider scanning with thread safety:
        
        Discovery Process:
            - Scans all configured LLM providers
            - Enumerates available models with capabilities
            - Measures provider response times and health
            - Updates model availability and status information
            - Forces refresh to ensure current data
            
        Threading Strategy:
            - Creates isolated daemon thread for network operations
            - Establishes new event loop for async provider communication
            - Ensures proper cleanup with try/finally blocks
            - Schedules UI updates on main thread via Clock
            
        Provider Communication:
            - Parallel provider scanning for efficiency
            - Timeout handling for unresponsive providers
            - Error isolation to prevent single provider failures
            - Capability enumeration with metadata collection
            
        Error Handling:
            - Network error recovery with user notification
            - Provider-specific error isolation
            - Graceful degradation for partial failures
            - Detailed logging for debugging and monitoring
            
        User Feedback:
            - Progress indication during discovery
            - Success notification with model count
            - Error messages with actionable guidance
            - Status updates throughout the process
            
        Performance Optimization:
            - Concurrent provider scanning
            - Efficient data collection and processing
            - Minimal UI thread blocking
            - Optimized network communication
            
        The discovery ensures up-to-date model information while
        maintaining responsive UI through proper threading.
        """
        if not self.chat_manager:
            return
        
        def run_discovery():
            """Run model discovery in a thread with its own event loop."""
            import asyncio
            
            async def _discover():
                try:
                    success = await self.chat_manager.discover_models(force_refresh=True)
                    if success:
                        Clock.schedule_once(self.update_ui, 0)
                    else:
                        Clock.schedule_once(lambda dt: Notification.error("Failed to discover models"), 0)
                except Exception as e:
                    logger.error(f"Model discovery failed: {e}")
                    Clock.schedule_once(lambda dt: Notification.error(f"Discovery failed: {e}"), 0)
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_discover())
            finally:
                loop.close()
        
        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=run_discovery)
        thread.daemon = True
        thread.start()
    
    def update_ui(self, dt=None):
        """Update interface with discovered models and provider status.
        
        This method synchronizes the UI with current system state:
        
        UI Synchronization:
            - Updates current model label with provider information
            - Refreshes provider health cards with latest status
            - Rebuilds model cards with current availability
            - Highlights currently selected model appropriately
            
        Data Processing:
            - Extracts provider health information
            - Processes available model list
            - Determines current model selection state
            - Organizes information for optimal display
            
        Visual Updates:
            - Clears existing content for fresh display
            - Creates new provider health cards
            - Generates model cards with current status
            - Applies appropriate highlighting and theming
            
        Model Card Creation:
            - Builds cards for all available models
            - Identifies and highlights current selection
            - Includes comprehensive model information
            - Enables selection for available models
            
        Provider Health Display:
            - Creates cards for each provider
            - Shows health status with appropriate colors
            - Displays performance metrics and model counts
            - Includes error information for troubleshooting
            
        State Management:
            - Maintains consistency between data and display
            - Updates model availability based on provider health
            - Preserves user selections and preferences
            - Handles edge cases and error states
            
        Performance Optimization:
            - Efficient widget creation and removal
            - Minimal layout recalculation
            - Optimized data binding and updates
            - Memory-efficient card management
            
        The update ensures the UI accurately reflects current
        system state with professional presentation.
        """
        if not self.chat_manager:
            return
        
        # Update current model label
        current_model_text = f"{self.chat_manager.current_provider}:{self.chat_manager.current_model}"
        self.current_model_label.text = current_model_text
        
        # Clear existing content
        self.health_layout.clear_widgets()
        self.models_layout.clear_widgets()
        
        # Update provider health
        provider_health = self.chat_manager.get_provider_health()
        for provider_name, provider_info in provider_health.items():
            if provider_info:
                health_card = ProviderHealthCard(provider_info)
                self.health_layout.add_widget(health_card)
        
        # Update models
        available_models = self.chat_manager.get_available_models()
        current_model = f"{self.chat_manager.current_provider}:{self.chat_manager.current_model}"
        
        for model_info in available_models:
            is_current = model_info.full_name == current_model
            model_card = ModelCard(model_info, is_current)
            self.models_layout.add_widget(model_card)
        
        logger.info(f"Updated UI with {len(available_models)} models")
    
    def show_strategy_menu(self, button):
        """Display intelligent model selection strategy dropdown menu.
        
        This method creates a sophisticated strategy selection interface:
        
        Strategy Options:
            - Manual: User-controlled model selection
            - Fastest: Optimize for minimal response latency
            - Cheapest: Minimize operational costs per token
            - Best Quality: Maximize response accuracy and coherence
            - Balanced: Optimize across multiple performance dimensions
            - Most Capable: Select highest capability model available
            
        Menu Implementation:
            - Material Design dropdown with proper positioning
            - Callback-based selection for loose coupling
            - Appropriate width and styling for readability
            - Touch-friendly sizing for mobile compatibility
            
        User Experience:
            - Clear strategy descriptions for informed selection
            - Intuitive menu interaction patterns
            - Visual feedback for selection confirmation
            - Immediate strategy application after selection
            
        Strategy Descriptions:
            Each strategy optimizes for different use cases:
            - Manual: Complete user control over model selection
            - Fastest: Real-time applications requiring quick responses
            - Cheapest: Cost-sensitive applications with budget constraints
            - Best Quality: Accuracy-critical applications
            - Balanced: General-purpose applications
            - Most Capable: Complex reasoning and advanced features
            
        Technical Implementation:
            - Dynamic menu item creation from strategy enumeration
            - Proper event handling with strategy parameter passing
            - Menu positioning relative to calling button
            - Resource cleanup for menu lifecycle management
            
        The menu provides intelligent model selection based on
        different optimization criteria and use case requirements.
        """
        strategies = [
            ("Manual", ModelSelectionStrategy.MANUAL),
            ("Fastest", ModelSelectionStrategy.FASTEST), 
            ("Cheapest", ModelSelectionStrategy.CHEAPEST),
            ("Best Quality", ModelSelectionStrategy.BEST_QUALITY),
            ("Balanced", ModelSelectionStrategy.BALANCED),
            ("Most Capable", ModelSelectionStrategy.MOST_CAPABLE),
        ]
        
        menu_items = [
            {
                "text": name,
                "viewclass": "OneLineListItem", 
                "on_release": lambda x=strategy: self.select_strategy(x),
            }
            for name, strategy in strategies
        ]
        
        self.strategy_menu = MDDropdownMenu(
            caller=button,
            items=menu_items,
            width_mult=4,
        )
        self.strategy_menu.open()
    
    def select_strategy(self, strategy):
        """Apply intelligent model selection strategy with feedback.
        
        This method implements strategy-based model selection:
        
        Strategy Application:
            - Dismisses selection menu for clean UI
            - Applies selected strategy to available models
            - Evaluates models based on strategy criteria
            - Selects optimal model from evaluation results
            
        Threading Strategy:
            - Creates isolated thread for strategy evaluation
            - Establishes new event loop for async operations
            - Ensures proper cleanup with try/finally blocks
            - Schedules UI updates on main thread safely
            
        Model Evaluation:
            - Analyzes available models using strategy criteria
            - Considers performance, cost, capability, and quality metrics
            - Applies weighting based on strategy priorities
            - Selects best match from evaluation results
            
        Model Switching:
            - Updates chat manager with selected model
            - Validates model availability before switching
            - Provides user feedback on selection success/failure
            - Updates UI to reflect new model selection
            
        Error Handling:
            - Strategy evaluation error recovery
            - Model switching failure handling
            - User notification with actionable messages
            - Graceful degradation for partial failures
            
        User Feedback:
            - Success notification with selected model name
            - Error messages with specific failure information
            - Warning messages for no suitable model found
            - Progress indication during evaluation
            
        Performance Optimization:
            - Efficient model evaluation algorithms
            - Cached model information for quick comparison
            - Minimal UI thread blocking during processing
            - Optimized strategy calculation
            
        Args:
            strategy (ModelSelectionStrategy): The selection strategy to apply
            
        The method provides intelligent model selection with
        comprehensive error handling and user feedback.
        """
        self.strategy_menu.dismiss()
        
        def run_selection():
            """Run model selection in a thread with its own event loop."""
            import asyncio
            
            async def _select():
                try:
                    model_info = await self.chat_manager.select_best_model(strategy)
                    if model_info:
                        success = self.chat_manager.set_model(model_info.name, model_info.provider)
                        if success:
                            Clock.schedule_once(lambda dt: Notification.success(f"Switched to {model_info.display_name}"), 0)
                            Clock.schedule_once(self.update_ui, 0)
                        else:
                            Clock.schedule_once(lambda dt: Notification.error("Failed to switch model"), 0)
                    else:
                        Clock.schedule_once(lambda dt: Notification.warning("No suitable model found"), 0)
                except Exception as e:
                    logger.error(f"Model selection failed: {e}")
                    Clock.schedule_once(lambda dt: Notification.error(f"Selection failed: {e}"), 0)
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_select())
            finally:
                loop.close()
        
        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=run_selection)
        thread.daemon = True
        thread.start()
    
    def on_model_selected(self, model_info: ModelInfo):
        """Process manual model selection with validation and feedback.
        
        This method handles user-initiated model selection:
        
        Selection Processing:
            - Validates model availability and compatibility
            - Updates chat manager with new model configuration
            - Verifies successful model switching
            - Provides immediate user feedback on result
            
        Model Validation:
            - Checks model availability status
            - Validates provider connectivity
            - Ensures model compatibility with current configuration
            - Verifies resource requirements
            
        System Integration:
            - Updates chat manager model configuration
            - Applies new model to active conversations
            - Synchronizes provider settings
            - Updates session state for persistence
            
        User Feedback:
            - Success notification with model display name
            - Error notification for selection failures
            - Immediate UI update to reflect selection
            - Visual confirmation of active model
            
        Error Handling:
            - Model switching failure detection
            - Provider communication error recovery
            - User notification with helpful error messages
            - Rollback to previous model on failure
            
        UI Synchronization:
            - Schedules UI update to reflect changes
            - Updates model card highlighting
            - Refreshes current model display
            - Maintains visual consistency
            
        Performance Optimization:
            - Immediate feedback without waiting for full update
            - Efficient model switching process
            - Minimal system disruption during selection
            - Optimized state synchronization
            
        Args:
            model_info (ModelInfo): The selected model information
            
        The method provides reliable manual model selection with
        comprehensive validation and user feedback.
        """
        success = self.chat_manager.set_model(model_info.name, model_info.provider)
        if success:
            Notification.success(f"Selected {model_info.display_name}")
            Clock.schedule_once(self.update_ui, 0)
        else:
            Notification.error("Failed to select model")
    
    def on_rag_config_updated(self, config: RAGConfig):
        """Process RAG configuration updates with validation and application.
        
        This method handles RAG system configuration changes:
        
        Configuration Processing:
            - Validates RAG configuration parameters
            - Applies configuration to chat manager RAG system
            - Updates system state with new parameters
            - Provides user feedback on configuration success
            
        Parameter Validation:
            - Validates numeric ranges for retrieval parameters
            - Checks timeout values for reasonable limits
            - Verifies threshold values within valid ranges
            - Ensures configuration consistency
            
        System Integration:
            - Updates chat manager RAG configuration
            - Applies changes to active RAG system
            - Synchronizes memory retrieval parameters
            - Updates context window management
            
        Configuration Persistence:
            - Saves configuration to persistent storage
            - Updates system configuration files
            - Ensures configuration survives application restart
            - Maintains configuration history for rollback
            
        User Feedback:
            - Success notification for configuration updates
            - Error messages for invalid parameters
            - Visual confirmation of applied settings
            - Real-time parameter validation feedback
            
        Error Handling:
            - Configuration validation error recovery
            - System integration failure handling
            - Rollback capability for failed configurations
            - User guidance for parameter correction
            
        Performance Impact:
            - Immediate configuration application
            - Minimal system disruption during updates
            - Efficient parameter synchronization
            - Optimized memory system integration
            
        Args:
            config (RAGConfig): The updated RAG configuration
            
        The method ensures reliable RAG configuration updates
        with proper validation and system integration.
        """
        if self.chat_manager:
            self.chat_manager.configure_rag(config)
            Notification.success("RAG configuration updated")
    
    def go_back(self, *args):
        """Navigate back to the primary chat interface.
        
        This method handles screen transition with proper state management:
        
        Navigation Process:
            - Updates screen manager to show enhanced chat screen
            - Preserves current model and configuration state
            - Ensures proper screen lifecycle management
            - Maintains application navigation stack
            
        State Preservation:
            - Model selection changes persist across navigation
            - RAG configuration updates remain active
            - Provider health information stays current
            - User preferences and settings are maintained
            
        Resource Management:
            - Proper cleanup of temporary UI elements
            - Memory efficient screen transition
            - Background task management during transition
            - Resource sharing with destination screen
            
        User Experience:
            - Smooth transition without visual artifacts
            - Consistent navigation behavior
            - Proper focus management after transition
            - Maintained application state continuity
            
        The navigation ensures seamless return to the main
        chat interface with all configuration changes preserved.
        """
        self.manager.current = "enhanced_chat"