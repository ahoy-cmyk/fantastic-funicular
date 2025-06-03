"""Provider configuration screen for setting up LLM providers."""

from typing import Any

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.selectioncontrol import MDSwitch
from kivymd.uix.textfield import MDTextField

from src.core.config import config_manager
from src.gui.theme import UIConstants
from src.gui.utils.notifications import Notification
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProviderConfigCard(MDCard):
    """Configuration card for a specific LLM provider."""

    def __init__(self, provider_name: str, **kwargs):
        super().__init__(**kwargs)
        self.provider_name = provider_name
        self.orientation = "vertical"
        self.padding = UIConstants.PADDING_MEDIUM
        self.spacing = UIConstants.SPACING_MEDIUM
        self.elevation = UIConstants.ELEVATION_CARD
        self.radius = [UIConstants.RADIUS_MEDIUM]
        self.size_hint_y = None
        self.adaptive_height = True

        self.build_ui()

    def build_ui(self):
        """Build the provider configuration UI."""
        # Header
        header = MDBoxLayout(
            orientation="horizontal",
            spacing=UIConstants.SPACING_MEDIUM,
            adaptive_height=True,
            size_hint_y=None,
        )

        # Provider info
        info_layout = MDBoxLayout(orientation="vertical", adaptive_height=True)

        # Title and status
        title_layout = MDBoxLayout(
            orientation="horizontal", spacing=UIConstants.SPACING_SMALL, adaptive_height=True
        )

        self.title_label = MDLabel(
            text=self.provider_name.title(),
            font_style="H6",
            theme_text_color="Primary",
            adaptive_height=True,
        )

        self.status_label = MDLabel(
            text="Not Configured",
            font_style="Caption",
            theme_text_color="Error",
            adaptive_height=True,
        )

        title_layout.add_widget(self.title_label)
        title_layout.add_widget(self.status_label)

        # Description
        descriptions = {
            "ollama": "Local LLM runner - free and private",
            "openai": "OpenAI-compatible API - supports OpenAI and other providers",
            "lmstudio": "Local API server - runs models locally",
        }

        desc_label = MDLabel(
            text=descriptions.get(self.provider_name, "LLM Provider"),
            font_style="Body2",
            theme_text_color="Secondary",
            adaptive_height=True,
        )

        info_layout.add_widget(title_layout)
        info_layout.add_widget(desc_label)

        # Enable switch
        self.enable_switch = MDSwitch(
            size_hint=(None, None), size=(dp(50), dp(30)), pos_hint={"center_y": 0.5}
        )
        self.enable_switch.bind(active=self.on_enable_changed)

        header.add_widget(info_layout)
        header.add_widget(self.enable_switch)

        # Configuration fields
        self.config_layout = MDBoxLayout(
            orientation="vertical", spacing=UIConstants.SPACING_SMALL, adaptive_height=True
        )

        self.add_widget(header)
        self.add_widget(self.config_layout)

        # Load current configuration
        Clock.schedule_once(lambda dt: self.load_config(), 0.1)

    def load_config(self):
        """Load current provider configuration."""
        try:
            # Get provider config with proper defaults
            if self.provider_name == "ollama":
                provider_config = {
                    "enabled": config_manager.get("providers.ollama_enabled", True),
                    "host": config_manager.get("providers.ollama_host", "http://localhost:11434"),
                }
                self.build_ollama_config(provider_config)
            elif self.provider_name == "openai":
                provider_config = {
                    "enabled": config_manager.get("providers.openai_enabled", False),
                    "api_key": config_manager.get("providers.openai_api_key") or "",
                    "base_url": config_manager.get("providers.openai_base_url") or "",
                    "organization": config_manager.get("providers.openai_organization") or "",
                }
                self.build_openai_config(provider_config)
            elif self.provider_name == "lmstudio":
                provider_config = {
                    "enabled": config_manager.get("providers.lmstudio_enabled", False),
                    "host": config_manager.get("providers.lmstudio_host", "http://localhost:1234"),
                }
                self.build_lmstudio_config(provider_config)

            # Update status
            self.update_status()

        except Exception as e:
            logger.error(f"Failed to load {self.provider_name} config: {e}")

    def build_ollama_config(self, provider_config: dict[str, Any]):
        """Build Ollama-specific configuration."""
        self.config_layout.clear_widgets()

        # Host URL
        self.host_field = MDTextField(
            hint_text="Host URL (e.g., http://localhost:11434)",
            text=provider_config.get("host", "http://localhost:11434"),
            helper_text="URL where Ollama is running",
            helper_text_mode="persistent",
        )
        self.host_field.bind(text=self.on_config_changed)
        self.config_layout.add_widget(self.host_field)

        # Available models section
        models_card = MDCard(
            orientation="vertical",
            padding=dp(15),
            spacing=dp(10),
            md_bg_color=(0.08, 0.08, 0.08, 1),
            adaptive_height=True,
            radius=[dp(8)],
        )

        models_title = MDLabel(
            text="Available Models",
            font_style="Subtitle1",
            theme_text_color="Primary",
            adaptive_height=True,
        )
        models_card.add_widget(models_title)

        # Model list and controls
        model_controls = MDBoxLayout(
            orientation="horizontal",
            spacing=dp(10),
            adaptive_height=True,
            size_hint_y=None,
            height=dp(50),
        )

        refresh_models_btn = MDRaisedButton(
            text="Refresh Models",
            on_release=self.refresh_ollama_models,
            size_hint_x=None,
            width=dp(120),
        )

        download_model_btn = MDRaisedButton(
            text="Download Model",
            on_release=self.download_ollama_model,
            size_hint_x=None,
            width=dp(120),
        )

        model_controls.add_widget(refresh_models_btn)
        model_controls.add_widget(download_model_btn)
        model_controls.add_widget(MDBoxLayout())  # Spacer

        models_card.add_widget(model_controls)

        # Models list
        self.ollama_models_list = MDLabel(
            text="Models will appear here after connecting...",
            theme_text_color="Secondary",
            font_style="Body2",
            adaptive_height=True,
            text_size=(None, None),
        )
        models_card.add_widget(self.ollama_models_list)

        self.config_layout.add_widget(models_card)

        # Test connection button
        test_btn = MDRaisedButton(text="Test Connection", on_release=self.test_ollama_connection)
        self.config_layout.add_widget(test_btn)

    def build_openai_config(self, provider_config: dict[str, Any]):
        """Build OpenAI-compatible API configuration."""
        self.config_layout.clear_widgets()

        # Base URL (for OpenAI-compatible APIs)
        self.base_url_field = MDTextField(
            hint_text="Base URL (optional, e.g., https://api.openai.com/v1)",
            text=provider_config.get("base_url", ""),
            helper_text="Leave empty for OpenAI, or enter custom API endpoint",
            helper_text_mode="persistent",
        )
        self.base_url_field.bind(text=self.on_config_changed)
        self.config_layout.add_widget(self.base_url_field)

        # API Key
        self.api_key_field = MDTextField(
            hint_text="API Key",
            text=provider_config.get("api_key", ""),
            password=True,
            helper_text="Your API key (sk-... for OpenAI, varies for others)",
            helper_text_mode="persistent",
        )
        self.api_key_field.bind(text=self.on_config_changed)
        self.config_layout.add_widget(self.api_key_field)

        # Organization (optional)
        self.org_field = MDTextField(
            hint_text="Organization ID (optional)",
            text=provider_config.get("organization", ""),
            helper_text="Optional organization ID (OpenAI specific)",
            helper_text_mode="persistent",
        )
        self.org_field.bind(text=self.on_config_changed)
        self.config_layout.add_widget(self.org_field)

        # Available models section
        models_card = MDCard(
            orientation="vertical",
            padding=dp(15),
            spacing=dp(10),
            md_bg_color=(0.08, 0.08, 0.08, 1),
            adaptive_height=True,
            radius=[dp(8)],
        )

        models_title = MDLabel(
            text="Available Models",
            font_style="Subtitle1",
            theme_text_color="Primary",
            adaptive_height=True,
        )
        models_card.add_widget(models_title)

        refresh_models_btn = MDRaisedButton(
            text="Refresh Models",
            on_release=self.refresh_openai_models,
            size_hint_x=None,
            width=dp(120),
        )
        models_card.add_widget(refresh_models_btn)

        self.openai_models_list = MDLabel(
            text="Models will appear here after connecting...",
            theme_text_color="Secondary",
            font_style="Body2",
            adaptive_height=True,
            text_size=(None, None),
        )
        models_card.add_widget(self.openai_models_list)

        self.config_layout.add_widget(models_card)

        # Test connection button
        test_btn = MDRaisedButton(text="Test Connection", on_release=self.test_openai_connection)
        self.config_layout.add_widget(test_btn)

    def build_lmstudio_config(self, provider_config: dict[str, Any]):
        """Build LM Studio-specific configuration."""
        self.config_layout.clear_widgets()

        # Host URL
        self.host_field = MDTextField(
            hint_text="LM Studio URL (e.g., http://localhost:1234)",
            text=provider_config.get("host", "http://localhost:1234"),
            helper_text="URL where LM Studio server is running",
            helper_text_mode="persistent",
        )
        self.host_field.bind(text=self.on_config_changed)
        self.config_layout.add_widget(self.host_field)

        # Installed models section
        models_card = MDCard(
            orientation="vertical",
            padding=dp(15),
            spacing=dp(10),
            md_bg_color=(0.08, 0.08, 0.08, 1),
            adaptive_height=True,
            radius=[dp(8)],
        )

        models_title = MDLabel(
            text="Installed Models",
            font_style="Subtitle1",
            theme_text_color="Primary",
            adaptive_height=True,
        )
        models_card.add_widget(models_title)

        refresh_models_btn = MDRaisedButton(
            text="Refresh Models",
            on_release=self.refresh_lmstudio_models,
            size_hint_x=None,
            width=dp(120),
        )
        models_card.add_widget(refresh_models_btn)

        self.lmstudio_models_list = MDLabel(
            text="Models will appear here after connecting...",
            theme_text_color="Secondary",
            font_style="Body2",
            adaptive_height=True,
            text_size=(None, None),
        )
        models_card.add_widget(self.lmstudio_models_list)

        self.config_layout.add_widget(models_card)

        # Test connection button
        test_btn = MDRaisedButton(text="Test Connection", on_release=self.test_lmstudio_connection)
        self.config_layout.add_widget(test_btn)

    def on_enable_changed(self, switch, value):
        """Handle enable/disable toggle."""
        try:
            config_manager.set_sync(f"providers.{self.provider_name}_enabled", value)
            self.update_status()

            if value:
                Notification.success(f"{self.provider_name.title()} enabled")
            else:
                Notification.info(f"{self.provider_name.title()} disabled")

        except Exception as e:
            logger.error(f"Failed to toggle {self.provider_name}: {e}")
            Notification.error("Failed to update provider settings")

    def on_config_changed(self, field, text):
        """Handle configuration field changes."""
        try:
            if self.provider_name == "ollama":
                config_manager.set_sync("providers.ollama_host", self.host_field.text)
            elif self.provider_name == "openai":
                config_manager.set_sync("providers.openai_api_key", self.api_key_field.text)
                if hasattr(self, "base_url_field"):
                    config_manager.set_sync("providers.openai_base_url", self.base_url_field.text)
                if hasattr(self, "org_field"):
                    config_manager.set_sync("providers.openai_organization", self.org_field.text)
            elif self.provider_name == "lmstudio":
                config_manager.set_sync("providers.lmstudio_host", self.host_field.text)

            self.update_status()

        except Exception as e:
            logger.error(f"Failed to save {self.provider_name} config: {e}")

    def update_status(self):
        """Update provider status indicator."""
        try:
            enabled = config_manager.get(f"providers.{self.provider_name}_enabled", False)

            if enabled:
                if self.provider_name == "ollama":
                    configured = bool(config_manager.get("providers.ollama_host"))
                elif self.provider_name == "openai":
                    configured = bool(config_manager.get("providers.openai_api_key"))
                elif self.provider_name == "lmstudio":
                    configured = bool(config_manager.get("providers.lmstudio_host"))
                else:
                    configured = False

                if configured:
                    self.status_label.text = "Configured"
                    self.status_label.theme_text_color = "Custom"
                    self.status_label.text_color = (0.2, 0.8, 0.4, 1.0)  # Green
                else:
                    self.status_label.text = "Enabled but not configured"
                    self.status_label.theme_text_color = "Custom"
                    self.status_label.text_color = (1.0, 0.6, 0.2, 1.0)  # Orange
            else:
                self.status_label.text = "Disabled"
                self.status_label.theme_text_color = "Secondary"

            self.enable_switch.active = enabled

        except Exception as e:
            logger.error(f"Failed to update {self.provider_name} status: {e}")

    def test_ollama_connection(self, *args):
        """Test Ollama connection."""
        Notification.info("Testing Ollama connection...")

        def run_test():
            try:
                import requests

                host = self.host_field.text or "http://localhost:11434"

                # Test connection with a simple request
                response = requests.get(f"{host}/api/tags", timeout=5)
                if response.status_code == 200:
                    Clock.schedule_once(
                        lambda dt: Notification.success("Ollama connection successful"), 0
                    )
                else:
                    Clock.schedule_once(
                        lambda dt: Notification.error(
                            f"Ollama connection failed: HTTP {response.status_code}"
                        ),
                        0,
                    )
            except requests.exceptions.ConnectionError:
                Clock.schedule_once(
                    lambda dt: Notification.error(
                        "Ollama connection failed: Cannot connect to server"
                    ),
                    0,
                )
            except requests.exceptions.Timeout:
                Clock.schedule_once(
                    lambda dt: Notification.error("Ollama connection failed: Request timeout"), 0
                )
            except Exception:
                Clock.schedule_once(
                    lambda dt: Notification.error(f"Ollama connection failed: {str(e)}"), 0
                )

        import threading

        thread = threading.Thread(target=run_test)
        thread.daemon = True
        thread.start()

    def test_openai_connection(self, *args):
        """Test OpenAI API key."""
        Notification.info("Testing OpenAI API key...")

        def run_test():
            try:
                import openai

                api_key = self.api_key_field.text
                organization = getattr(self, "org_field", None)
                org_id = organization.text if organization else None

                if not api_key:
                    Clock.schedule_once(lambda dt: Notification.error("API key is required"), 0)
                    return

                # Create client and test with a simple request
                client = openai.OpenAI(api_key=api_key, organization=org_id if org_id else None)

                # Test with a simple models list request
                models = client.models.list()
                Clock.schedule_once(lambda dt: Notification.success("OpenAI API key is valid"), 0)

            except openai.AuthenticationError:
                Clock.schedule_once(lambda dt: Notification.error("OpenAI API key is invalid"), 0)
            except openai.PermissionDeniedError:
                Clock.schedule_once(
                    lambda dt: Notification.error("OpenAI API key lacks required permissions"), 0
                )
            except Exception:
                Clock.schedule_once(
                    lambda dt: Notification.error(f"OpenAI connection failed: {str(e)}"), 0
                )

        import threading

        thread = threading.Thread(target=run_test)
        thread.daemon = True
        thread.start()

    def test_lmstudio_connection(self, *args):
        """Test LM Studio connection."""
        Notification.info("Testing LM Studio connection...")

        def run_test():
            try:
                import requests

                host = self.host_field.text or "http://localhost:1234"

                # Test connection with models endpoint
                response = requests.get(f"{host}/v1/models", timeout=5)
                if response.status_code == 200:
                    models = response.json()
                    if models.get("data"):
                        Clock.schedule_once(
                            lambda dt: Notification.success(
                                f"LM Studio connection successful - {len(models['data'])} models available"
                            ),
                            0,
                        )
                    else:
                        Clock.schedule_once(
                            lambda dt: Notification.warning(
                                "LM Studio connected but no models loaded"
                            ),
                            0,
                        )
                else:
                    Clock.schedule_once(
                        lambda dt: Notification.error(
                            f"LM Studio connection failed: HTTP {response.status_code}"
                        ),
                        0,
                    )
            except requests.exceptions.ConnectionError:
                Clock.schedule_once(
                    lambda dt: Notification.error(
                        "LM Studio connection failed: Cannot connect to server"
                    ),
                    0,
                )
            except requests.exceptions.Timeout:
                Clock.schedule_once(
                    lambda dt: Notification.error("LM Studio connection failed: Request timeout"), 0
                )
            except Exception:
                Clock.schedule_once(
                    lambda dt: Notification.error(f"LM Studio connection failed: {str(e)}"), 0
                )

        import threading

        thread = threading.Thread(target=run_test)
        thread.daemon = True
        thread.start()

    # Model management methods
    def refresh_ollama_models(self, *args):
        """Refresh Ollama models list."""
        Notification.info("Refreshing Ollama models...")

        def run_refresh():
            try:
                import requests

                host = self.host_field.text or "http://localhost:11434"

                response = requests.get(f"{host}/api/tags", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])

                    if models:
                        model_text = "\n".join(
                            [
                                f"• {model.get('name', 'Unknown')} ({model.get('size', 'Unknown size')})"
                                for model in models
                            ]
                        )
                        Clock.schedule_once(
                            lambda dt: self._update_ollama_models(
                                f"Found {len(models)} models:\n{model_text}"
                            ),
                            0,
                        )
                    else:
                        Clock.schedule_once(
                            lambda dt: self._update_ollama_models(
                                "No models found. Use 'Download Model' to add models."
                            ),
                            0,
                        )
                else:
                    Clock.schedule_once(
                        lambda dt: self._update_ollama_models(
                            "Failed to fetch models. Check connection."
                        ),
                        0,
                    )

            except Exception:
                Clock.schedule_once(lambda dt: self._update_ollama_models(f"Error: {str(e)}"), 0)

        import threading

        thread = threading.Thread(target=run_refresh)
        thread.daemon = True
        thread.start()

    def _update_ollama_models(self, text):
        """Update Ollama models display."""
        if hasattr(self, "ollama_models_list"):
            self.ollama_models_list.text = text

    def download_ollama_model(self, *args):
        """Download a new Ollama model."""
        try:
            from kivymd.uix.boxlayout import MDBoxLayout
            from kivymd.uix.button import MDFlatButton
            from kivymd.uix.dialog import MDDialog
            from kivymd.uix.textfield import MDTextField

            # Create input dialog
            content = MDBoxLayout(
                orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(120)
            )

            self.model_name_field = MDTextField(
                hint_text="Model name (e.g., llama3.2, mistral, codellama)",
                helper_text="Enter the model name to download from Ollama library",
                helper_text_mode="persistent",
            )
            content.add_widget(self.model_name_field)

            self.download_dialog = MDDialog(
                title="Download Ollama Model",
                type="custom",
                content_cls=content,
                buttons=[
                    MDFlatButton(
                        text="Cancel", on_release=lambda x: self.download_dialog.dismiss()
                    ),
                    MDFlatButton(text="Download", on_release=self._start_model_download),
                ],
                size_hint=(0.8, None),
                height=dp(300),
            )
            self.download_dialog.open()

        except Exception as e:
            logger.error(f"Error creating download dialog: {e}")
            Notification.error("Failed to open download dialog")

    def _start_model_download(self, *args):
        """Start downloading the specified model."""
        try:
            model_name = self.model_name_field.text.strip()
            if not model_name:
                Notification.error("Please enter a model name")
                return

            # Close dialog
            self.download_dialog.dismiss()

            # Start download in background
            Notification.info(f"Starting download of {model_name}...")

            def download_model():
                try:
                    import requests

                    host = self.host_field.text or "http://localhost:11434"

                    # Start the download
                    response = requests.post(
                        f"{host}/api/pull",
                        json={"name": model_name},
                        timeout=300,  # 5 minutes timeout for download start
                    )

                    if response.status_code == 200:
                        Clock.schedule_once(
                            lambda dt: Notification.success(
                                f"Successfully downloaded {model_name}"
                            ),
                            0,
                        )
                        # Refresh models list
                        Clock.schedule_once(lambda dt: self.refresh_ollama_models(), 1)
                    else:
                        Clock.schedule_once(
                            lambda dt: Notification.error(
                                f"Download failed: HTTP {response.status_code}"
                            ),
                            0,
                        )

                except requests.exceptions.Timeout:
                    Clock.schedule_once(
                        lambda dt: Notification.error(
                            "Download timeout - large models may take longer"
                        ),
                        0,
                    )
                except Exception:
                    Clock.schedule_once(
                        lambda dt: Notification.error(f"Download failed: {str(e)}"), 0
                    )

            import threading

            thread = threading.Thread(target=download_model)
            thread.daemon = True
            thread.start()

        except Exception as e:
            logger.error(f"Error starting model download: {e}")
            Notification.error("Failed to start download")

    def refresh_openai_models(self, *args):
        """Refresh OpenAI models list."""
        Notification.info("Refreshing OpenAI models...")

        def run_refresh():
            try:
                import openai

                api_key = self.api_key_field.text
                base_url = getattr(self, "base_url_field", None)
                org_id = getattr(self, "org_field", None)

                if not api_key:
                    Clock.schedule_once(
                        lambda dt: self._update_openai_models("API key required"), 0
                    )
                    return

                client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url.text if base_url and base_url.text else None,
                    organization=org_id.text if org_id and org_id.text else None,
                )

                models = client.models.list()
                model_names = [model.id for model in models.data]

                if model_names:
                    model_text = "\n".join([f"• {name}" for name in sorted(model_names)])
                    Clock.schedule_once(
                        lambda dt: self._update_openai_models(f"Available models:\n{model_text}"), 0
                    )
                else:
                    Clock.schedule_once(
                        lambda dt: self._update_openai_models("No models available"), 0
                    )

            except Exception:
                Clock.schedule_once(lambda dt: self._update_openai_models(f"Error: {str(e)}"), 0)

        import threading

        thread = threading.Thread(target=run_refresh)
        thread.daemon = True
        thread.start()

    def _update_openai_models(self, text):
        """Update OpenAI models display."""
        if hasattr(self, "openai_models_list"):
            self.openai_models_list.text = text

    def refresh_lmstudio_models(self, *args):
        """Refresh LM Studio models list."""
        Notification.info("Refreshing LM Studio models...")

        def run_refresh():
            try:
                import requests

                host = self.host_field.text or "http://localhost:1234"

                response = requests.get(f"{host}/v1/models", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])

                    if models:
                        model_text = "\n".join(
                            [f"• {model.get('id', 'Unknown')}" for model in models]
                        )
                        Clock.schedule_once(
                            lambda dt: self._update_lmstudio_models(
                                f"Loaded models:\n{model_text}"
                            ),
                            0,
                        )
                    else:
                        Clock.schedule_once(
                            lambda dt: self._update_lmstudio_models(
                                "No models loaded in LM Studio"
                            ),
                            0,
                        )
                else:
                    Clock.schedule_once(
                        lambda dt: self._update_lmstudio_models(
                            "Failed to fetch models. Check connection."
                        ),
                        0,
                    )

            except Exception:
                Clock.schedule_once(lambda dt: self._update_lmstudio_models(f"Error: {str(e)}"), 0)

        import threading

        thread = threading.Thread(target=run_refresh)
        thread.daemon = True
        thread.start()

    def _update_lmstudio_models(self, text):
        """Update LM Studio models display."""
        if hasattr(self, "lmstudio_models_list"):
            self.lmstudio_models_list.text = text


class ProviderConfigScreen(MDScreen):
    """Screen for configuring LLM providers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()

    def build_ui(self):
        """Build the provider configuration UI."""
        # Main layout
        main_layout = MDBoxLayout(orientation="vertical")

        # Top bar - custom toolbar
        toolbar = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(56),
            md_bg_color=(0.12, 0.12, 0.12, 1),
            padding=[dp(16), 0, dp(16), 0],
        )

        # Back button
        back_btn = MDFlatButton(text="Back", on_release=lambda x: self.go_back())
        toolbar.add_widget(back_btn)

        # Title
        title_label = MDLabel(
            text="LLM Provider Configuration", font_style="H6", theme_text_color="Primary"
        )
        toolbar.add_widget(title_label)

        # Spacer
        toolbar.add_widget(MDBoxLayout())

        # Action buttons
        refresh_btn = MDFlatButton(text="Refresh", on_release=lambda x: self.refresh_config())
        save_btn = MDRaisedButton(text="Save All", on_release=lambda x: self.save_all())

        toolbar.add_widget(refresh_btn)
        toolbar.add_widget(save_btn)

        # Scroll content
        scroll = MDScrollView()
        content = MDBoxLayout(
            orientation="vertical",
            padding=UIConstants.PADDING_LARGE,
            spacing=UIConstants.SPACING_LARGE,
            adaptive_height=True,
        )

        # Instructions
        instructions = MDLabel(
            text="Configure your LLM providers below. Enable at least one provider to start chatting with AI.",
            font_style="Body1",
            theme_text_color="Secondary",
            adaptive_height=True,
            text_size=(None, None),
        )
        content.add_widget(instructions)

        # Provider cards
        providers = ["ollama", "openai", "lmstudio"]
        for provider in providers:
            card = ProviderConfigCard(provider)
            content.add_widget(card)

        scroll.add_widget(content)

        main_layout.add_widget(toolbar)
        main_layout.add_widget(scroll)

        self.add_widget(main_layout)

    def go_back(self):
        """Navigate back to settings."""
        self.manager.current = "settings"

    def refresh_config(self):
        """Refresh configuration from disk."""
        try:
            config_manager.reload()
            Notification.success("Configuration refreshed")

            # Rebuild UI
            self.clear_widgets()
            self.build_ui()

        except Exception as e:
            logger.error(f"Failed to refresh config: {e}")
            Notification.error("Failed to refresh configuration")

    def save_all(self):
        """Save all provider configurations."""
        try:
            config_manager.save()

            # Refresh providers in chat manager (if available)
            try:

                # Note: This creates a new instance, but in a real app you'd want to refresh the existing one
                # For now, providers will be refreshed when chat manager is next created
                pass
            except Exception as e:
                logger.warning(f"Could not refresh chat manager providers: {e}")

            Notification.success("All provider settings saved")

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            Notification.error("Failed to save configuration")
