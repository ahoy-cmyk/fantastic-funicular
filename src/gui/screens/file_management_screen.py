"""File management screen for RAG documents."""

import asyncio
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDIconButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel
from kivymd.uix.list import MDList, ThreeLineAvatarIconListItem
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.textfield import MDTextField

from src.gui.theme import UIConstants
from src.gui.utils.notifications import Notification
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FileCard(MDCard):
    """Card displaying uploaded file information and controls."""
    
    def __init__(self, file_data: Dict, on_delete_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.file_data = file_data
        self.on_delete_callback = on_delete_callback
        self.orientation = "vertical"
        self.padding = UIConstants.PADDING_MEDIUM
        self.spacing = UIConstants.SPACING_SMALL
        self.elevation = UIConstants.ELEVATION_CARD
        self.radius = [UIConstants.RADIUS_MEDIUM]
        self.size_hint_y = None
        self.adaptive_height = True
        
        self.build_ui()
    
    def build_ui(self):
        """Build file card UI."""
        # Header with file name and actions
        header = MDBoxLayout(
            orientation="horizontal",
            spacing=UIConstants.SPACING_MEDIUM,
            adaptive_height=True,
            size_hint_y=None
        )
        
        # File info
        info_layout = MDBoxLayout(orientation="vertical", adaptive_height=True)
        
        # File name
        name_label = MDLabel(
            text=self.file_data.get('file_name', 'Unknown File'),
            font_style="Subtitle1",
            theme_text_color="Primary",
            adaptive_height=True,
            bold=True
        )
        
        # File details
        file_size = self.file_data.get('file_size', 0)
        size_str = self._format_file_size(file_size)
        file_type = self.file_data.get('file_type', 'unknown')
        upload_date = self.file_data.get('upload_date', 'Unknown')
        
        details_label = MDLabel(
            text=f"Type: {file_type.upper()} • Size: {size_str} • Uploaded: {upload_date}",
            font_style="Caption",
            theme_text_color="Secondary",
            adaptive_height=True
        )
        
        # Content preview
        content = self.file_data.get('content', '')
        preview = content[:200] + "..." if len(content) > 200 else content
        preview_label = MDLabel(
            text=f"Preview: {preview}",
            font_style="Body2",
            theme_text_color="Secondary",
            adaptive_height=True
        )
        
        info_layout.add_widget(name_label)
        info_layout.add_widget(details_label)
        info_layout.add_widget(preview_label)
        
        # Action buttons
        actions_layout = MDBoxLayout(
            orientation="horizontal",
            spacing=UIConstants.SPACING_SMALL,
            adaptive_height=True,
            size_hint_x=None,
            width=dp(100)
        )
        
        # View button
        view_button = MDIconButton(
            icon="eye",
            on_release=self.view_file
        )
        
        # Delete button
        delete_button = MDIconButton(
            icon="delete",
            theme_icon_color="Custom",
            icon_color=(1, 0.3, 0.3, 1),
            on_release=self.delete_file
        )
        
        actions_layout.add_widget(view_button)
        actions_layout.add_widget(delete_button)
        
        header.add_widget(info_layout)
        header.add_widget(actions_layout)
        
        self.add_widget(header)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    def view_file(self, *args):
        """Show file content in a dialog."""
        content = self.file_data.get('content', 'No content available')
        
        content_layout = MDBoxLayout(
            orientation="vertical",
            spacing=dp(10),
            size_hint_y=None,
            height=dp(400)
        )
        
        # Create scrollable text field for content
        content_field = MDTextField(
            text=content,
            multiline=True,
            readonly=True,
            size_hint_y=None,
            height=dp(350)
        )
        
        content_layout.add_widget(content_field)
        
        dialog = MDDialog(
            title=f"View: {self.file_data.get('file_name', 'File')}",
            type="custom",
            content_cls=content_layout,
            buttons=[
                MDFlatButton(text="Close", on_release=lambda x: dialog.dismiss())
            ]
        )
        dialog.open()
    
    def delete_file(self, *args):
        """Delete the file with confirmation."""
        if self.on_delete_callback:
            self.on_delete_callback(self.file_data)


class FileManagementScreen(MDScreen):
    """Screen for managing uploaded RAG files."""
    
    def __init__(self, chat_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.name = "file_management"
        self.chat_manager = chat_manager
        self.uploaded_files = []
        
        self.build_ui()
        
        # Load existing files
        Clock.schedule_once(self.load_files, 1.0)
    
    def build_ui(self):
        """Build the file management UI."""
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
            text="File Management",
            font_style="H4",
            theme_text_color="Primary",
            adaptive_height=True
        )
        
        # Add file button
        add_button = MDRaisedButton(
            text="Add File",
            icon="plus",
            size_hint_y=None,
            height=dp(40),
            on_release=self.show_file_upload
        )
        
        # Refresh button
        refresh_button = MDIconButton(
            icon="refresh",
            on_release=self.load_files
        )
        
        header.add_widget(back_button)
        header.add_widget(title)
        header.add_widget(add_button)
        header.add_widget(refresh_button)
        
        main_layout.add_widget(header)
        
        # File count info
        self.file_count_label = MDLabel(
            text="Loading files...",
            font_style="Subtitle1",
            theme_text_color="Secondary",
            adaptive_height=True,
            size_hint_y=None
        )
        main_layout.add_widget(self.file_count_label)
        
        # Scrollable file list
        scroll = MDScrollView()
        self.files_layout = MDBoxLayout(
            orientation="vertical",
            spacing=UIConstants.SPACING_MEDIUM,
            adaptive_height=True,
            size_hint_y=None
        )
        scroll.add_widget(self.files_layout)
        main_layout.add_widget(scroll)
        
        self.add_widget(main_layout)
    
    def load_files(self, dt=None):
        """Load uploaded files from memory system."""
        def run_load():
            async def _load():
                try:
                    if not self.chat_manager or not hasattr(self.chat_manager, 'memory_manager'):
                        Clock.schedule_once(lambda dt: self._update_files_ui([]), 0)
                        return
                    
                    # Get all memories and filter for uploaded files
                    all_memories = await self.chat_manager.memory_manager.get_all_memories(limit=1000)
                    files = [m for m in all_memories if m.metadata and m.metadata.get("source") == "file_upload"]
                    
                    file_data = []
                    for memory in files:
                        metadata = memory.metadata or {}
                        if metadata.get("source") == "file_upload":
                            file_info = {
                                'id': memory.id,
                                'file_name': metadata.get('file_name', 'Unknown'),
                                'file_size': metadata.get('file_size', 0),
                                'file_type': metadata.get('file_type', 'unknown'),
                                'upload_date': memory.created_at.strftime('%Y-%m-%d %H:%M') if memory.created_at else 'Unknown',
                                'content': memory.content[:1000] if memory.content else '',  # Limit content for display
                                'memory': memory
                            }
                            file_data.append(file_info)
                    
                    Clock.schedule_once(lambda dt: self._update_files_ui(file_data), 0)
                    
                except Exception as e:
                    error_msg = f"Failed to load files: {str(e)}"
                    logger.error(error_msg)
                    Clock.schedule_once(lambda dt: Notification.error(error_msg), 0)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_load())
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_load)
        thread.daemon = True
        thread.start()
    
    def _update_files_ui(self, files):
        """Update the UI with loaded files."""
        self.uploaded_files = files
        
        # Clear existing cards
        self.files_layout.clear_widgets()
        
        # Update count
        self.file_count_label.text = f"Files: {len(files)}"
        
        if not files:
            # Show empty state
            empty_label = MDLabel(
                text="No files uploaded yet.\nClick 'Add File' to upload documents for RAG.",
                font_style="Body1",
                theme_text_color="Secondary",
                adaptive_height=True,
                halign="center"
            )
            self.files_layout.add_widget(empty_label)
        else:
            # Show file cards
            for file_data in files:
                file_card = FileCard(
                    file_data=file_data,
                    on_delete_callback=self.confirm_delete_file
                )
                self.files_layout.add_widget(file_card)
    
    def show_file_upload(self, *args):
        """Show file upload dialog."""
        def file_manager_open():
            self.file_manager = MDFileManager(
                exit_manager=self.exit_file_manager,
                select_path=self.select_file_path,
                ext=['.txt', '.pdf', '.md', '.json', '.py', '.js', '.java', '.cpp', '.c', '.html', '.xml', '.csv', '.docx']
            )
            self.file_manager.show('/')
        
        # Create file upload dialog
        content = MDBoxLayout(
            orientation="vertical",
            spacing=dp(10),
            size_hint_y=None,
            height=dp(200)
        )
        
        info_label = MDLabel(
            text="Upload a file to add its content to the RAG memory system.\n"
                 "Supported formats: txt, pdf, md, json, py, js, java, cpp, c, html, xml, csv, docx",
            theme_text_color="Primary",
            adaptive_height=True
        )
        
        browse_button = MDRaisedButton(
            text="Browse Files",
            size_hint_y=None,
            height=dp(40),
            on_release=lambda x: (dialog.dismiss(), file_manager_open())
        )
        
        content.add_widget(info_label)
        content.add_widget(browse_button)
        
        dialog = MDDialog(
            title="Upload File for RAG",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda x: dialog.dismiss())
            ]
        )
        
        dialog.open()
    
    def exit_file_manager(self, *args):
        """Exit the file manager."""
        self.file_manager.close()
    
    def select_file_path(self, path):
        """Handle file selection for upload."""
        self.exit_file_manager()
        
        async def process_file():
            try:
                file_size = os.path.getsize(path)
                
                # Check file size (limit to 10MB)
                if file_size > 10 * 1024 * 1024:
                    Clock.schedule_once(lambda dt: Notification.warning("File too large (max 10MB)"), 0)
                    return
                
                file_ext = os.path.splitext(path)[1].lower()
                content = ""
                
                try:
                    if file_ext == '.pdf':
                        # Extract text from PDF
                        from pypdf import PdfReader
                        reader = PdfReader(path)
                        content = ""
                        for page in reader.pages:
                            content += page.extract_text() + "\n"
                    elif file_ext in ['.docx']:
                        Clock.schedule_once(lambda dt: Notification.info("DOCX support coming soon"), 0)
                        return
                    else:
                        # Text-based files
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                except Exception as e:
                    error_msg = f"Error reading file: {str(e)}"
                    Clock.schedule_once(lambda dt: Notification.error(error_msg), 0)
                    return
                
                # Store in memory with metadata
                file_name = os.path.basename(path)
                metadata = {
                    "source": "file_upload",
                    "file_name": file_name,
                    "file_path": path,
                    "file_size": file_size,
                    "file_type": file_ext,
                    "upload_date": datetime.now().isoformat()
                }
                
                # Add to memory system
                memory_id = await self.chat_manager.memory_manager.remember(
                    content=f"File content from {file_name}:\n\n{content}",
                    auto_classify=True,
                    metadata=metadata
                )
                
                if memory_id:
                    Clock.schedule_once(lambda dt: Notification.success(f"File '{file_name}' added to RAG memory"), 0)
                    Clock.schedule_once(self.load_files, 0)  # Reload files
                    logger.info(f"Uploaded file to RAG: {file_name} ({file_size} bytes)")
                else:
                    Clock.schedule_once(lambda dt: Notification.error("Failed to add file to memory"), 0)
                    
            except Exception as e:
                error_msg = f"Upload failed: {str(e)}"
                logger.error(f"File upload error: {e}")
                Clock.schedule_once(lambda dt: Notification.error(error_msg), 0)
        
        # Run in thread
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(process_file())
            loop.close()
        
        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()
    
    def confirm_delete_file(self, file_data):
        """Show confirmation dialog for deleting a file."""
        content = MDBoxLayout(
            orientation="vertical",
            spacing=dp(10),
            size_hint_y=None,
            height=dp(100)
        )
        
        warning_label = MDLabel(
            text=f"Delete '{file_data.get('file_name', 'Unknown')}'?\nThis action cannot be undone.",
            theme_text_color="Primary",
            adaptive_height=True,
            halign="center"
        )
        content.add_widget(warning_label)
        
        dialog = MDDialog(
            title="Delete File",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="Delete",
                    md_bg_color=(0.8, 0.2, 0.2, 1),
                    on_release=lambda x: self.delete_file(file_data, dialog)
                )
            ]
        )
        dialog.open()
    
    def delete_file(self, file_data, dialog):
        """Delete a file from the memory system."""
        dialog.dismiss()
        
        def run_delete():
            async def _delete():
                try:
                    memory = file_data.get('memory')
                    if memory and hasattr(self.chat_manager, 'memory_manager'):
                        await self.chat_manager.memory_manager.forget(memory.id)
                        Clock.schedule_once(lambda dt: Notification.success("File deleted"), 0)
                        Clock.schedule_once(self.load_files, 0)  # Reload files
                    else:
                        Clock.schedule_once(lambda dt: Notification.error("Failed to delete file"), 0)
                except Exception as e:
                    error_msg = f"Delete failed: {str(e)}"
                    logger.error(f"Failed to delete file: {e}")
                    Clock.schedule_once(lambda dt: Notification.error(error_msg), 0)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_delete())
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_delete)
        thread.daemon = True
        thread.start()
    
    def go_back(self, *args):
        """Navigate back to the chat screen."""
        self.manager.current = "enhanced_chat"