"""Model management screen for selecting and configuring models."""

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
    """Card displaying model information and controls."""
    
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
    """Card showing provider health status."""
    
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
    """Card for configuring RAG system."""
    
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
    """Screen for managing models and RAG configuration."""
    
    def __init__(self, chat_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.name = "model_management"
        self.chat_manager = chat_manager
        self.current_model_info = None
        
        self.build_ui()
        
        # Schedule model discovery
        Clock.schedule_once(self.discover_models, 1.0)
    
    def build_ui(self):
        """Build the model management UI."""
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
        """Discover available models."""
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
        """Update UI with discovered models and provider health."""
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
        """Show model selection strategy menu."""
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
        """Select and apply model selection strategy."""
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
        """Handle manual model selection."""
        success = self.chat_manager.set_model(model_info.name, model_info.provider)
        if success:
            Notification.success(f"Selected {model_info.display_name}")
            Clock.schedule_once(self.update_ui, 0)
        else:
            Notification.error("Failed to select model")
    
    def on_rag_config_updated(self, config: RAGConfig):
        """Handle RAG configuration updates."""
        if self.chat_manager:
            self.chat_manager.configure_rag(config)
            Notification.success("RAG configuration updated")
    
    def go_back(self, *args):
        """Navigate back to the chat screen."""
        self.manager.current = "enhanced_chat"