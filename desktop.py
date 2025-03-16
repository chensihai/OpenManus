#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenManus Desktop Application
A Windows desktop application for the OpenManus agent using GTK.
"""

# Platform detection
import os
import sys
import logging
import threading
import tomli
import tomli_w  # For writing TOML files
import asyncio
import ctypes
from typing import Optional, Dict, Any, List

# GUI toolkit imports
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
from gi.repository import Gio

# Environment setup
from dotenv import load_dotenv
load_dotenv(override=True)  # Force reload environment variables from .env file

# OpenManus imports
from app.agent.manus import Manus
from app.logger import logger

# Windows-specific configuration
if os.name == 'nt':
    # Force basic theme and backend
    os.environ["GTK_THEME"] = "win32"
    os.environ["GDK_BACKEND"] = "win32"

    # High DPI fix
    ctypes.windll.shcore.SetProcessDpiAwareness(1)


class GtkLogHandler(logging.Handler):
    """Custom log handler that redirects log messages to a GTK TextView"""

    def __init__(self, text_view):
        super().__init__()
        self.text_view = text_view
        self.buffer = text_view.get_buffer()
        self.formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')

    def emit(self, record):
        log_entry = self.formatter.format(record) + '\n'

        # Use GLib.idle_add to safely update the UI from any thread
        GLib.idle_add(self.update_text_view, log_entry)

    def update_text_view(self, text):
        end_iter = self.buffer.get_end_iter()
        self.buffer.insert(end_iter, text)
        # Auto-scroll to the end
        self.text_view.scroll_to_iter(self.buffer.get_end_iter(), 0.0, False, 0.0, 0.0)
        return False  # Required for GLib.idle_add


class ManusApp(Gtk.Application):
    """
    Main application class for OpenManus desktop application.
    Manages application-level functionality, configuration, and client initialization.
    """
    def __init__(self):
        super().__init__(application_id='com.openmanus.desktop')

        # Initialize configuration parameters
        self.config_params = {}

        # Initialize agent
        self.agent = None

        # Initialize background tasks
        self.running_tasks = []

    def do_activate(self):
        """Create main window when application is activated"""
        self.main_window = MainWindow(self)
        self.main_window.show_all()
        self.main_window.present()  # Force focus

    def initialize_agent(self):
        """Initialize the AI agent with current settings"""
        if not hasattr(self, 'main_window') or self.main_window is None:
            logger.error("Main window not initialized")
            return

        provider = self.main_window.provider_combo.get_active_text().lower()
        model = self.main_window.model_combo.get_active_text()

        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.toml')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file missing: {config_path}")

            with open(config_path, 'rb') as f:
                config = tomli.load(f)

            # Config structure validation
            if 'llm' not in config:
                config['llm'] = {}
            if 'default' not in config['llm']:
                # If there are direct settings in llm, move them to default
                default_settings = {}
                direct_keys = ['model', 'base_url', 'api_key', 'max_tokens', 'temperature', 'api_type', 'api_version']

                for key in direct_keys:
                    if key in config['llm']:
                        default_settings[key] = config['llm'][key]
                        del config['llm'][key]

                # Add missing required fields
                required_fields = {
                    'model': 'gpt-3.5-turbo',
                    'base_url': 'https://api.openai.com/v1',
                    'api_key': '',
                    'max_tokens': 2048,
                    'temperature': 0.7,
                    'api_type': 'Openai',
                    'api_version': ''
                }

                for field, default_value in required_fields.items():
                    if field not in default_settings or default_settings[field] is None:
                        default_settings[field] = default_value

                config['llm']['default'] = default_settings
                modified = True
            else:
                # Ensure required fields exist in llm.default
                required_fields = {
                    'model': 'gpt-3.5-turbo',
                    'base_url': 'https://api.openai.com/v1',
                    'api_key': '',
                    'max_tokens': 2048,
                    'temperature': 0.7,
                    'api_type': 'Openai',
                    'api_version': ''
                }

                for field, default_value in required_fields.items():
                    if field not in config['llm']['default'] or config['llm']['default'][field] is None:
                        config['llm']['default'][field] = default_value
                        modified = True

            if modified:
                with open(config_path, 'wb') as f:
                    tomli_w.dump(config, f)
                logging.info("Updated config structure")

        except Exception as e:
            logging.error(f"Error ensuring config fields: {str(e)}")

        self.agent = Manus(
            provider=provider,
            api_key=config['llm']['default']['api_key'],
            model=model,
            temperature=self.main_window.temp_scale.get_value(),
            max_tokens=int(self.main_window.tokens_scale.get_value()),
            base_url=config['llm']['default']['base_url'],
            api_version=config['llm']['default']['api_version']
        )

        # Vision configuration
        if config.get('vision', {}).get('enabled', False):
            self.agent.enable_vision(
                model=config['vision'].get('model', 'gpt-4-vision-preview'),
                max_tokens=config['vision'].get('max_tokens', 2048)
            )
            self.main_window.vision_switch.set_active(True)

        # Initialize tool lists
        available_tools = []
        enabled_tools = []

        if hasattr(self.agent, 'enable_tools'):
            available_tools = self.agent.get_available_tools()
            enabled_tools = [
                t.lower().replace(' ', '_')
                for t in config.get('tools', {}).get('enabled', [])
                if t.lower().replace(' ', '_') in available_tools
            ]
            self.agent.enable_tools(enabled_tools)

            # Update UI checkboxes with display names
            for display_name, checkbox in self.main_window.tool_switches:
                tool_id = display_name.lower().replace(' ', '_')
                checkbox.set_active(tool_id in enabled_tools)
                checkbox.set_visible(tool_id in available_tools)

        # Initialize UI state
        self.main_window.model_combo.set_active_id(config['llm']['default']['model'])
        self.main_window.temp_scale.set_value(config['llm']['default']['temperature'])
        self.main_window.tokens_scale.set_value(config['llm']['default']['max_tokens'])

        for tool, checkbox in self.main_window.tool_switches:
            checkbox.set_active(tool in enabled_tools)
            checkbox.set_visible(tool in available_tools)

    def run_background_task(self, task_function, callback, *args, **kwargs):
        """Run a task in the background to keep UI responsive"""
        def run_async_task():
            # Create a new event loop for the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Run the coroutine in the new event loop
                result = loop.run_until_complete(task_function(*args, **kwargs))
                # Schedule the callback to run in the main thread
                GLib.idle_add(lambda: callback(result))
            except Exception as e:
                # Schedule error handling in the main thread
                GLib.idle_add(lambda: callback(None, error=str(e)))
            finally:
                loop.close()

        # Start the thread
        thread = threading.Thread(target=run_async_task)
        thread.daemon = True
        thread.start()
        self.running_tasks.append(thread)
        return thread


class MainWindow(Gtk.ApplicationWindow):
    """
    Main window class for OpenManus desktop application.
    Handles the UI components, user interactions, and display logic.
    """
    def __init__(self, app):
        super().__init__(application=app, title="OpenManus")
        self.app = app
        self.set_default_size(1200, 800)
        self.set_size_request(800, 600)  # Minimum size constraints

        # Header bar with menu
        self.build_header_bar()

        # Main content
        self.build_content_area()
        self.connect("destroy", Gtk.main_quit)

    def build_header_bar(self):
        header = Gtk.HeaderBar()
        header.set_show_close_button(True)
        header.set_title("OpenManus Desktop")

        # Menu button with stacked icon
        menu_button = Gtk.MenuButton()
        menu_icon = Gtk.Image.new_from_icon_name("open-menu-symbolic", Gtk.IconSize.BUTTON)
        menu_button.set_image(menu_icon)

        # Menu items
        menu = Gio.Menu()
        menu.append("About OpenManus", "win.about")
        menu.append("Documentation", "win.docs")
        menu.append("Preferences", "win.preferences")
        menu.append("Keyboard Shortcuts", "win.shortcuts")
        menu.append("Privacy Policy", "win.privacy")

        # Create action group
        self.action_group = Gio.SimpleActionGroup()
        actions = [
            ('about', None, self.on_about),
            ('docs', None, self.on_docs),
            ('preferences', None, self.on_preferences),
            ('shortcuts', None, self.on_shortcuts),
            ('privacy', None, self.on_privacy)
        ]
        for name, param, callback in actions:
            action = Gio.SimpleAction.new(name, param)
            action.connect('activate', callback)
            self.action_group.add_action(action)

        self.insert_action_group("win", self.action_group)
        menu_button.set_menu_model(menu)
        header.pack_end(menu_button)
        self.set_titlebar(header)

    def build_content_area(self):
        main_paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        main_paned.set_position(300)  # 25% of 1200px default width

        # Left Settings Panel
        settings_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        settings_box.set_margin_top(12)
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(6)

        # AI Provider Configuration
        provider_frame = self.create_config_frame("AI Provider")
        provider_grid = Gtk.Grid(column_spacing=6, row_spacing=6)
        
        # Provider Selection
        self.provider_combo = Gtk.ComboBoxText()
        self.provider_combo.append_text("OpenAI")
        self.provider_combo.append_text("Anthropic")
        provider_grid.attach(self.create_label("Provider:"), 0, 0, 1, 1)
        provider_grid.attach(self.provider_combo, 1, 0, 1, 1)

        # API Key
        self.api_key_entry = Gtk.Entry()
        self.api_key_entry.set_visibility(False)
        provider_grid.attach(self.create_label("API Key:"), 0, 1, 1, 1)
        provider_grid.attach(self.api_key_entry, 1, 1, 1, 1)

        # API URL
        self.api_url_entry = Gtk.Entry()
        provider_grid.attach(self.create_label("API URL:"), 0, 2, 1, 1)
        provider_grid.attach(self.api_url_entry, 1, 2, 1, 1)

        provider_frame.add(provider_grid)
        settings_box.pack_start(provider_frame, False, False, 0)

        # Model Parameters
        model_frame = self.create_config_frame("Model Parameters")
        model_grid = Gtk.Grid(column_spacing=6, row_spacing=6)
        
        self.temperature_scale = self.create_scale("Temperature", 0, 2, 0.1)
        self.max_tokens_spin = self.create_spin("Max Tokens", 128, 8192, 512)
        self.top_p_scale = self.create_scale("Top P", 0, 1, 0.1)
        self.frequency_penalty_scale = self.create_scale("Frequency Penalty", 0, 2, 0.1)
        
        model_grid.attach(self.temperature_scale, 0, 0, 1, 1)
        model_grid.attach(self.max_tokens_spin, 0, 1, 1, 1)
        model_grid.attach(self.top_p_scale, 0, 2, 1, 1)
        model_grid.attach(self.frequency_penalty_scale, 0, 3, 1, 1)
        model_frame.add(model_grid)

        settings_box.pack_start(model_frame, True, True, 0)

        # Chat Interface
        chat_box = self.build_chat_interface()
        
        main_paned.pack1(settings_box, resize=False, shrink=False)
        main_paned.pack2(chat_box, resize=True, shrink=False)
        
        self.add(main_paned)

    def create_config_frame(self, label_text, margin=6):
        frame = Gtk.Frame()
        frame.set_label(label_text)
        frame.set_label_align(0.1, 0.5)
        frame.set_margin_top(margin)
        frame.set_margin_bottom(margin)
        frame.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
        return frame

    def create_label(self, text):
        label = Gtk.Label(label=text)
        label.set_halign(Gtk.Align.START)
        label.set_margin_end(6)
        return label

    def load_config_into_ui(self):
        try:
            config = self.load_config()
            provider = config.get('provider', 'OpenAI')
            self.provider_combo.set_active_id(provider)
            
            # Load provider-specific settings
            provider_config = config.get(provider.lower(), {})
            self.api_key_entry.set_text(provider_config.get('api_key', ''))
            self.api_url_entry.set_text(provider_config.get('base_url', ''))
            self.temperature_scale.set_value(provider_config.get('temperature', 1.0))
            self.max_tokens_spin.set_value(provider_config.get('max_tokens', 2000))
            
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_config_from_ui(self):
        config = self.load_config()
        provider = self.provider_combo.get_active_text()
        
        config['provider'] = provider
        config[provider.lower()] = {
            'api_key': self.api_key_entry.get_text(),
            'base_url': self.api_url_entry.get_text(),
            'temperature': self.temperature_scale.get_value(),
            'max_tokens': self.max_tokens_spin.get_value(),
            'top_p': self.top_p_scale.get_value(),
            'frequency_penalty': self.frequency_penalty_scale.get_value()
        }
        
        self.save_config(config)

    def load_config(self):
        try:
            with open('config/config.toml', 'rb') as f:
                return tomli.load(f)
        except FileNotFoundError:
            return {}

    def save_config(self, config):
        with open('config/config.toml', 'wb') as f:
            tomli_w.dump(config, f)

    def on_about(self, action, param):
        """Show about dialog"""
        dialog = Gtk.AboutDialog(transient_for=self, modal=True)
        dialog.set_program_name("OpenManus Desktop")
        dialog.set_version("1.0")
        dialog.set_comments("A desktop application for the OpenManus agent")
        dialog.set_website("https://openmanus.com")
        dialog.set_website_label("OpenManus Website")
        dialog.set_authors(["Your Name"])
        dialog.set_copyright("Copyright 2023 Your Name")
        dialog.run()
        dialog.destroy()

    def on_docs(self, action, param):
        """Show documentation"""
        print("Documentation")

    def on_preferences(self, action, param):
        """Show preferences dialog"""
        print("Preferences")

    def on_shortcuts(self, action, param):
        """Show keyboard shortcuts"""
        print("Shortcuts")

    def on_privacy(self, action, param):
        """Show privacy policy"""
        print("Privacy Policy")

    def migrate_legacy_config(self):
        """Ensure config has required fields"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.toml')
            if not os.path.exists(config_path):
                # Create basic config file with correct structure
                basic_config = {
                    'llm': {
                        'default': {
                            'model': 'gpt-3.5-turbo',
                            'base_url': 'https://api.openai.com/v1',
                            'api_key': '',
                            'max_tokens': 4096,
                            'temperature': 1.0,
                            'api_type': 'Openai',
                            'api_version': ''
                        }
                    }
                }

                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'wb') as f:
                    tomli_w.dump(basic_config, f)
                return

            # Read existing config
            with open(config_path, 'rb') as f:
                config = tomli.load(f)

            modified = False

            # Ensure llm section exists
            if 'llm' not in config:
                config['llm'] = {}
                modified = True

            # Ensure llm.default section exists
            if 'default' not in config['llm']:
                # If there are direct settings in llm, move them to default
                default_settings = {}
                direct_keys = ['model', 'base_url', 'api_key', 'max_tokens', 'temperature', 'api_type', 'api_version']

                for key in direct_keys:
                    if key in config['llm']:
                        default_settings[key] = config['llm'][key]
                        del config['llm'][key]

                # Add missing required fields
                required_fields = {
                    'model': 'gpt-4o',
                    'base_url': 'https://api.openai.com/v1',
                    'api_key': '',
                    'max_tokens': 4096,
                    'temperature': 1.0,
                    'api_type': 'Openai',
                    'api_version': ''
                }

                for field, default_value in required_fields.items():
                    if field not in default_settings or default_settings[field] is None:
                        default_settings[field] = default_value

                config['llm']['default'] = default_settings
                modified = True
            else:
                # Ensure required fields exist in llm.default
                required_fields = {
                    'model': 'gpt-4o',
                    'base_url': 'https://api.openai.com/v1',
                    'api_key': '',
                    'max_tokens': 4096,
                    'temperature': 1.0,
                    'api_type': 'Openai',
                    'api_version': ''
                }

                for field, default_value in required_fields.items():
                    if field not in config['llm']['default'] or config['llm']['default'][field] is None:
                        config['llm']['default'][field] = default_value
                        modified = True

            if modified:
                with open(config_path, 'wb') as f:
                    tomli_w.dump(config, f)
                logging.info("Updated config structure")

        except Exception as e:
            logging.error(f"Error ensuring config fields: {str(e)}")

    def on_model_changed(self, combo):
        """Handle model selection change"""
        model = combo.get_active_text()
        logger.info(f"Model changed to: {model}")
        # Update config
        if self.app.config_params.get('llm') is None:
            self.app.config_params['llm'] = {}
        self.app.config_params['llm']['default']['model'] = model

    def on_provider_changed(self, combo):
        """Handle provider selection change"""
        active_text = combo.get_active_text()
        if active_text is None:
            # No provider selected, add default provider
            self.provider_combo.handler_block_by_func(self.on_provider_changed)
            self.provider_combo.remove_all()
            self.provider_combo.append_text("OpenAI")
            self.provider_combo.set_active(0)
            self.provider_combo.handler_unblock_by_func(self.on_provider_changed)
            return

        provider = active_text.lower()
        models = self._get_provider_models(provider)
        self.model_combo.handler_block_by_func(self.on_model_changed)
        self.model_combo.remove_all()
        if models:
            for model in models:
                self.model_combo.append_text(model)
            # Set first model as default if available
            self.model_combo.set_active(0)
            logger.info(f"Loaded {len(models)} models for {provider}")
        else:
            logger.warning(f"No models found for {provider}, using defaults")
            default_models = ["gpt-3.5-turbo", "gpt-4"]
            for model in default_models:
                self.model_combo.append_text(model)
            self.model_combo.set_active(0)
        self.model_combo.handler_unblock_by_func(self.on_model_changed)

    def _get_provider_models(self, provider):
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.toml')
            with open(config_path, 'rb') as f:
                config = tomli.load(f)

            # Handle legacy config format
            if 'providers' not in config:
                # Convert legacy format to new providers format
                if provider == 'openai' and 'llm' in config:
                    # Return models from llm section if available
                    if 'model' in config['llm']:
                        return [config['llm']['model']]
                    else:
                        return ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
                else:
                    # Default models for other providers
                    default_models = {
                        'anthropic': ["claude-3-opus", "claude-3-sonnet"],
                        'mistral': ["mistral-large", "mistral-medium"],
                        'groq': ["llama3-70b", "llama3-8b"],
                        'ollama': ["llama3", "mistral"]
                    }
                    return default_models.get(provider, [])

            # New config format
            if provider in config['providers']:
                return config['providers'][provider].get('models', [])
            return []

        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return []

    def on_temperature_changed(self, scale):
        """Handle temperature setting change"""
        temp = scale.get_value()
        logger.info(f"Temperature changed to: {temp}")
        # Update config
        if self.app.config_params.get('llm') is None:
            self.app.config_params['llm'] = {}
        self.app.config_params['llm']['default']['temperature'] = temp

    def on_tokens_changed(self, scale):
        """Handle max tokens setting change"""
        tokens = int(scale.get_value())
        logger.info(f"Max tokens changed to: {tokens}")
        # Update config
        if self.app.config_params.get('llm') is None:
            self.app.config_params['llm'] = {'default': {}}
        self.app.config_params['llm']['default']['max_tokens'] = tokens

    def on_tool_toggled(self, checkbox, tool_name):
        """Handle tool checkbox toggle"""
        active = checkbox.get_active()
        logger.info(f"Tool '{tool_name}' {'enabled' if active else 'disabled'}")
        # Update config (to be implemented)

    def on_settings_clicked(self, widget):
        """Update settings dialog for current provider and show it"""
        current_provider = self.provider_combo.get_active_text()
        self.update_settings_for_provider(current_provider)
        self.settings_dialog.show_all()

    def update_settings_for_provider(self, provider_name):
        """Update settings dialog fields based on selected provider"""
        # Update labels
        self.api_key_label.set_text(f"{provider_name} API Key:")

        # Load settings
        try:
            with open('config/config.toml', 'rb') as f:
                config = tomli.load(f)

            if 'llm' in config:
                self.api_key_entry.set_text(config['llm'].get('default', {}).get('api_key', ''))
                self.base_url_entry.set_text(config['llm'].get('default', {}).get('base_url', self.get_default_base_url(provider_name.lower())))

            if 'browser' in config:
                if 'headless' in config['browser']:
                    self.headless_switch.set_active(config['browser']['headless'])
                if 'chrome_instance_path' in config['browser']:
                    self.chrome_path_entry.set_text(config['browser']['chrome_instance_path'])
                if 'wss_url' in config['browser']:
                    self.wss_url_entry.set_text(config['browser']['wss_url'])
                if 'cdp_url' in config['browser']:
                    self.cdp_url_entry.set_text(config['browser']['cdp_url'])
        except Exception as e:
            logging.error(f"Error loading settings: {str(e)}")

    def get_default_base_url(self, provider):
        """Return default base URL for known providers"""
        defaults = {
            'openai': 'https://api.openai.com/v1',
            'anthropic': 'https://api.anthropic.com',
            'mistral': 'https://api.mistral.ai/v1',
            'groq': 'https://api.groq.com/v1',
            'ollama': 'http://localhost:11434/api'
        }
        return defaults.get(provider, '')

    def on_save_settings(self, button):
        """Save settings"""
        provider = self.provider_combo.get_active_text().lower()

        try:
            # Load existing config
            with open('config/config.toml', 'rb') as f:
                config = tomli.load(f)

            # Update llm section
            if 'llm' not in config:
                config['llm'] = {}

            config['llm']['default']['api_key'] = self.api_key_entry.get_text()
            base_url = self.api_url_entry.get_text()
            if base_url:
                config['llm']['default']['base_url'] = base_url

            # Add defaults if they don't exist
            if 'model' not in config['llm']['default']:
                config['llm']['default']['model'] = 'gpt-4o'
            if 'max_tokens' not in config['llm']['default']:
                config['llm']['default']['max_tokens'] = 4096
            if 'temperature' not in config['llm']['default']:
                config['llm']['default']['temperature'] = 1.0
            if 'api_type' not in config['llm']['default']:
                config['llm']['default']['api_type'] = 'Openai'
            if 'api_version' not in config['llm']['default']:
                config['llm']['default']['api_version'] = ''

            # Update browser section
            if self.headless_switch.get_active() or self.chrome_path_entry.get_text() or self.wss_url_entry.get_text() or self.cdp_url_entry.get_text():
                if 'browser' not in config:
                    config['browser'] = {}

                config['browser']['headless'] = self.headless_switch.get_active()

                chrome_path = self.chrome_path_entry.get_text()
                if chrome_path:
                    config['browser']['chrome_instance_path'] = chrome_path

                wss_url = self.wss_url_entry.get_text()
                if wss_url:
                    config['browser']['wss_url'] = wss_url

                cdp_url = self.cdp_url_entry.get_text()
                if cdp_url:
                    config['browser']['cdp_url'] = cdp_url

            # Save config
            with open('config/config.toml', 'wb') as f:
                tomli_w.dump(config, f)

            self.settings_dialog.hide()
        except Exception as e:
            logging.error(f"Error saving settings: {str(e)}")
            self.show_error_dialog(f"Failed to save settings: {str(e)}")

    def on_run_clicked(self, button):
        """Handle run button click"""
        # Initialize agent if not already initialized
        if self.app.agent is None:
            self.app.initialize_agent()
            self.status_bar.push(self.status_context, "Agent initialized")

    def on_submit_clicked(self, button):
        # Get config overrides from UI
        vision_enabled = self.vision_switch.get_active()

        # Update agent config before execution
        if hasattr(self.app.agent, 'vision_enabled'):
            self.app.agent.vision_enabled = vision_enabled

        # Get selected tools from UI
        tools = [tool[0] for tool in self.tool_switches if tool[1].get_active()]

        # Get prompt from text view
        prompt = self.prompt_text.get_buffer().get_text(
            self.prompt_text.get_buffer().get_start_iter(),
            self.prompt_text.get_buffer().get_end_iter(),
            True
        )

        if not prompt.strip():
            self.show_error_dialog("Please enter a prompt")
            return

        # Clear response
        self.response_text.get_buffer().set_text("")

        # Log the start of agent execution to the response area
        logging.info("Starting agent execution with prompt: %s", prompt[:50] + "..." if len(prompt) > 50 else prompt)

        # Start progress indicator
        self.start_progress_pulse()

        # Initialize agent if needed
        if not hasattr(self.app, 'agent') or self.app.agent is None:
            self.app.initialize_agent()

        # Get values directly from UI widgets
        model = self.model_combo.get_active_text() or "gpt-3.5-turbo"
        temperature = self.temp_scale.get_value()
        max_tokens = int(self.tokens_scale.get_value())

        # Run agent in background
        self.submit_button.set_sensitive(False)
        self._run_agent_thread(prompt, model, temperature, max_tokens, tools)

    async def _run_agent_async(self, prompt, model, temperature, max_tokens, tools):
        """Run the agent asynchronously and capture its output"""
        try:
            # Configure the agent if attributes exist
            if hasattr(self.app.agent, 'model'):
                self.app.agent.model = model
            if hasattr(self.app.agent, 'temperature'):
                self.app.agent.temperature = temperature
            if hasattr(self.app.agent, 'max_tokens'):
                self.app.agent.max_tokens = max_tokens

            # Log configuration
            logging.info(f"Agent configuration: model={model}, temp={temperature}, max_tokens={max_tokens}")
            logging.info(f"Selected tools: {tools}")

            # Run the agent
            response = await self.app.agent.run(prompt)

            # Log and display the final response
            logging.info(f"Agent response: {response}")

            # Update UI on the main thread
            GLib.idle_add(self.update_response_complete, response)

        except Exception as e:
            error_msg = f"Error running agent: {str(e)}"
            logging.error(error_msg)
            GLib.idle_add(self.update_response_error, error_msg)
        finally:
            # Always re-enable the submit button
            GLib.idle_add(self.submit_button.set_sensitive, True)
            GLib.idle_add(self.stop_progress_pulse)

    def _run_agent_thread(self, prompt, model, temperature, max_tokens, tools):
        """Run the agent in a separate thread"""
        # Create and run the asyncio event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_agent_async(prompt, model, temperature, max_tokens, tools))
        finally:
            loop.close()

    def update_response_complete(self, response):
        """Update the UI with the completed response"""
        # Add a separator line
        buffer = self.response_text.get_buffer()
        end_iter = buffer.get_end_iter()
        buffer.insert(end_iter, "\n\n----- FINAL RESPONSE -----\n\n")
        buffer.insert(end_iter, response)

        # Update status
        self.status_bar.push(self.status_context, "Request completed")
        return False  # Required for GLib.idle_add

    def update_response_error(self, error_msg):
        """Update the UI with an error message"""
        buffer = self.response_text.get_buffer()
        end_iter = buffer.get_end_iter()
        buffer.insert(end_iter, f"\n\nERROR: {error_msg}\n")

        # Update status
        self.status_bar.push(self.status_context, "Error occurred")
        return False  # Required for GLib.idle_add

    def start_progress_pulse(self):
        """Start progress bar pulsing animation"""
        self.progress_pulse_active = True

        def update_pulse():
            if self.progress_pulse_active:
                self.progress_bar.pulse()
                return True
            else:
                self.progress_bar.set_fraction(0.0)
                return False

        self.progress_pulse_id = GLib.timeout_add(100, update_pulse)

    def stop_progress_pulse(self):
        """Stop progress bar pulsing animation"""
        self.progress_pulse_active = False

    def show_error_dialog(self, message):
        """Show an error dialog with the given message"""
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Error"
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def on_close(self, widget, event):
        """Handle window close event"""
        # Clean up resources
        return False  # Allow the window to close

    def build_chat_interface(self):
        chat_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        
        # Prompt input
        prompt_label = self.create_label("Your Prompt:")
        self.prompt_buffer = Gtk.TextBuffer()
        self.prompt_text = Gtk.TextView(buffer=self.prompt_buffer)
        self.prompt_text.set_wrap_mode(Gtk.WrapMode.WORD)
        prompt_scroll = Gtk.ScrolledWindow()
        prompt_scroll.set_min_content_height(80)  # ~3 lines at 27px/line
        prompt_scroll.add(self.prompt_text)

        # Response display
        response_label = self.create_label("Agent Response:")
        self.response_buffer = Gtk.TextBuffer()
        self.response_view = Gtk.TextView(buffer=self.response_buffer)
        self.response_view.set_editable(False)
        self.response_view.set_wrap_mode(Gtk.WrapMode.WORD)
        response_scroll = Gtk.ScrolledWindow()
        response_scroll.set_min_content_height(200)
        response_scroll.add(self.response_view)

        # Control buttons
        self.submit_btn = Gtk.Button(label="Submit Query")
        self.submit_btn.connect("clicked", self.on_submit_clicked)

        # Progress indicator
        self.progress_bar = Gtk.ProgressBar()

        chat_box.pack_start(prompt_label, False, False, 0)
        chat_box.pack_start(prompt_scroll, True, True, 0)
        chat_box.pack_start(self.submit_btn, False, False, 5)
        chat_box.pack_start(response_label, False, False, 0)
        chat_box.pack_start(response_scroll, True, True, 0)
        chat_box.pack_start(self.progress_bar, False, False, 0)

        return chat_box

    def create_scale(self, label_text, min_val, max_val, step):
        box = Gtk.Box(spacing=6)
        label = self.create_label(label_text)
        adj = Gtk.Adjustment(value=1.0, lower=min_val, upper=max_val, step_increment=step)
        scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj)
        scale.set_digits(2)
        scale.set_hexpand(True)
        box.pack_start(label, False, False, 0)
        box.pack_start(scale, True, True, 0)
        return box

    def create_spin(self, label_text, min_val, max_val, step):
        box = Gtk.Box(spacing=6)
        label = self.create_label(label_text)
        spin = Gtk.SpinButton.new_with_range(min_val, max_val, step)
        box.pack_start(label, False, False, 0)
        box.pack_start(spin, True, True, 0)
        return box

    def on_provider_changed(self, combo):
        provider = combo.get_active_text()
        self.update_provider_visibility(provider)

    def update_provider_visibility(self, provider):
        # Update UI visibility based on provider
        if provider == "OpenAI":
            self.api_key_label.set_visible(True)
            self.api_key_entry.set_visible(True)
            self.base_url_label.set_visible(True)
            self.base_url_entry.set_visible(True)
        elif provider == "Anthropic":
            self.api_key_label.set_visible(True)
            self.api_key_entry.set_visible(True)
            self.base_url_label.set_visible(False)
            self.base_url_entry.set_visible(False)
        else:
            self.api_key_label.set_visible(False)
            self.api_key_entry.set_visible(False)
            self.base_url_label.set_visible(False)
            self.base_url_entry.set_visible(False)


class SettingsDialog(Gtk.Dialog):
    """Settings dialog for configuring API keys and other settings"""
    def __init__(self, parent):
        super().__init__(
            title="Settings",
            transient_for=parent,
            flags=Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT
        )
        self.parent = parent
        self.set_default_size(400, 300)

        # Add buttons
        self.add_button("Cancel", Gtk.ResponseType.CANCEL)
        self.add_button("Save", Gtk.ResponseType.OK)

        # Build UI
        self.build_ui()

    def build_ui(self):
        """Build the settings dialog UI"""
        content_area = self.get_content_area()
        content_area.set_margin_top(10)
        content_area.set_margin_bottom(10)
        content_area.set_margin_start(10)
        content_area.set_margin_end(10)
        content_area.set_spacing(10)

        # API Key settings
        api_label = Gtk.Label(label="OpenAI API Key:")
        api_label.set_halign(Gtk.Align.START)
        content_area.pack_start(api_label, False, False, 0)

        self.api_entry = Gtk.Entry()
        self.api_entry.set_visibility(False)  # Hide API key
        self.api_entry.set_placeholder_text("Enter your OpenAI API key")
        content_area.pack_start(self.api_entry, False, False, 0)

        # Base URL settings
        url_label = Gtk.Label(label="API Base URL:")
        url_label.set_halign(Gtk.Align.START)
        content_area.pack_start(url_label, False, False, 10)

        self.url_entry = Gtk.Entry()
        self.url_entry.set_placeholder_text("https://api.openai.com/v1")
        content_area.pack_start(self.url_entry, False, False, 0)

        # Load existing settings if available
        self.load_settings()

        # Show all widgets
        self.show_all()

    def load_settings(self):
        """Load existing settings into the dialog"""
        # Get settings from app config
        config = self.parent.app.config_params.get('llm', {})

        # Set API key if available
        if 'default' in config and 'api_key' in config['default']:
            self.api_entry.set_text(config['default']['api_key'])

        # Set base URL if available
        if 'default' in config and 'base_url' in config['default']:
            self.url_entry.set_text(config['default']['base_url'])


def main():
    """Main entry point for the desktop application"""
    app = ManusApp()
    app.hold()  # Prevent garbage collection
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
