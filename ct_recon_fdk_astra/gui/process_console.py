import os
import sys
import shlex
import time
import re
import html
from PyQt6.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser, QPushButton, QLabel, QFrame, QFileDialog, QProgressBar
)
from PyQt6.QtCore import QProcess, pyqtSignal, Qt, QTimer, QProcessEnvironment
from PyQt6.QtGui import QFont

class AnsiHtmlParser:
    """
    A stateful parser that converts ANSI escape sequences to valid HTML fragments
    and scans plain text for clickable URLs and existing local file paths.
    """
    def __init__(self):
        self.bold = False
        self.fg_color = None
        self.bg_color = None
        self.span_open = False
        
        # Mapping for standard 16 ANSI colors
        self.ansi_colors = {
            '30': '#000000', '31': '#ff5555', '32': '#50fa7b', '33': '#f1fa8c',
            '34': '#bd93f9', '35': '#ff79c6', '36': '#8be9fd', '37': '#f8f8f2',
            '90': '#6272a4', '91': '#ff6e6e', '92': '#69ff94', '93': '#ffffa5',
            '94': '#d6acff', '95': '#ff92df', '96': '#a4ffff', '97': '#ffffff'
        }

    def get_8bit_color(self, idx: int) -> str:
        if idx < 16:
            ansi_16 = [
                '#000000', '#800000', '#008000', '#808000', '#000080', '#800080', '#008080', '#c0c0c0',
                '#808080', '#ff0000', '#00ff00', '#ffff00', '#0000ff', '#ff00ff', '#00ffff', '#ffffff'
            ]
            return ansi_16[idx]
        elif idx < 232:
            idx -= 16
            b = idx % 6
            g = (idx // 6) % 6
            r = (idx // 36) % 6
            return f"rgb({r*51},{g*51},{b*51})"
        else:
            val = 8 + (idx - 232) * 10
            return f"rgb({val},{val},{val})"

    def build_span_start(self) -> str:
        styles = []
        if self.bold:
            styles.append("font-weight: bold;")
        if self.fg_color:
            styles.append(f"color: {self.fg_color};")
        if self.bg_color:
            styles.append(f"background-color: {self.bg_color};")
        if styles:
            return f"<span style='{'; '.join(styles)}'>"
        return ""

    def parse(self, text: str, linkify_files: bool = True, working_dir: str = None) -> str:
        # Strip non-SGR escape sequences (like Erase Line \x1b[K, cursor show/hide, etc.)
        text = re.sub(r'\x1b\[[0-9;?]*[a-ln-zA-Z]', '', text)
        parts = re.split(r'\x1b\[(.*?)m', text)
        html_parts = []
        
        current_span_style = self.build_span_start() if self.span_open else ""
        if current_span_style:
            html_parts.append(current_span_style)
            
        for i, part in enumerate(parts):
            if i % 2 == 1:
                # ANSI code sequence
                codes = part.split(';')
                idx = 0
                while idx < len(codes):
                    code = codes[idx]
                    if not code or code == '0':
                        self.bold = False
                        self.fg_color = None
                        self.bg_color = None
                    elif code == '1':
                        self.bold = True
                    elif code in self.ansi_colors:
                        self.fg_color = self.ansi_colors[code]
                    elif code == '38':
                        if idx + 2 < len(codes):
                            mode = codes[idx+1]
                            if mode == '5':
                                self.fg_color = self.get_8bit_color(int(codes[idx+2]))
                                idx += 2
                            elif mode == '2' and idx + 4 < len(codes):
                                r = codes[idx+2]
                                g = codes[idx+3]
                                b = codes[idx+4]
                                self.fg_color = f"rgb({r},{g},{b})"
                                idx += 4
                    elif code == '39':
                        self.fg_color = None
                    elif code == '49':
                        self.bg_color = None
                    idx += 1
            else:
                # Text content
                if not part:
                    continue
                
                linkified = self.linkify(part, linkify_files, working_dir)
                
                new_span_style = self.build_span_start()
                if new_span_style != current_span_style:
                    if current_span_style:
                        html_parts.append("</span>")
                    if new_span_style:
                        html_parts.append(new_span_style)
                        self.span_open = True
                    else:
                        self.span_open = False
                    current_span_style = new_span_style
                    
                html_parts.append(linkified)
                
        if current_span_style:
            html_parts.append("</span>")
            
        final_html = "".join(html_parts)
        return final_html.replace("\n", "<br>").replace("\r", "<br>")

    def linkify(self, text: str, linkify_files: bool, working_dir: str) -> str:
        # Pre-process links and file paths using placeholders to prevent HTML entity collisions.
        placeholders = {}
        placeholder_idx = 0
        
        # 1. URL pattern
        url_pattern = r'(https?://[a-zA-Z0-9\-\.\_\~\:\/\?\#\[\]\@\!\$\&\'\(\)\*\+\,;\=\%]+)'
        
        def url_repl(match):
            nonlocal placeholder_idx
            url = match.group(1)
            
            # Clean trailing punctuation
            stripped_url = url
            suffix = ""
            while stripped_url and stripped_url[-1] in ".,!?;:'\"()[]{}":
                suffix = stripped_url[-1] + suffix
                stripped_url = stripped_url[:-1]
                
            placeholder = f"___URL_PLACEHOLDER_{placeholder_idx}___"
            escaped_url = html.escape(stripped_url)
            placeholders[placeholder] = f'<a href="{escaped_url}" style="color: #89b4fa; text-decoration: underline;">{escaped_url}</a>'
            placeholder_idx += 1
            return placeholder + suffix
            
        text_with_placeholders = re.sub(url_pattern, url_repl, text)
        
        if linkify_files:
            # 2. File path pattern (using lookbehind instead of \b to correctly support leading slashes)
            path_pattern = r'((?<![a-zA-Z0-9_\-\./])(?:\.\./|\./|/)[a-zA-Z0-9_\-\./]+)'
            
            def path_repl(match):
                nonlocal placeholder_idx
                path = match.group(1)
                
                # Strip trailing punctuation
                stripped_path = path
                suffix = ""
                while stripped_path and stripped_path[-1] in ".,!?;:":
                    suffix = stripped_path[-1] + suffix
                    stripped_path = stripped_path[:-1]
                    
                check_path = stripped_path
                if not os.path.isabs(check_path) and working_dir:
                    check_path = os.path.join(working_dir, stripped_path)
                if os.path.exists(check_path):
                    abs_path = os.path.abspath(check_path)
                    placeholder = f"___PATH_PLACEHOLDER_{placeholder_idx}___"
                    escaped_path = html.escape(stripped_path)
                    escaped_abs_path = html.escape(abs_path)
                    placeholders[placeholder] = f'<a href="revealfile://{escaped_abs_path}" style="color: #a6e3a1; text-decoration: underline;">{escaped_path}</a>'
                    placeholder_idx += 1
                    return placeholder + suffix
                return path
                
            text_with_placeholders = re.sub(path_pattern, path_repl, text_with_placeholders)
            
        # 3. HTML-escape the plain text with placeholders
        escaped_text = html.escape(text_with_placeholders)
        
        # 4. Swap placeholders back with their HTML links
        for placeholder, link_html in placeholders.items():
            escaped_text = escaped_text.replace(placeholder, link_html)
            
        return escaped_text


class ProcessConsoleWindow(QDialog):
    """
    A reusable dialog that executes an external command in a PyQt6 QProcess,
    displaying colored ANSI and rich console logs with clickable links (URLs/files).
    
    Supports optional real-time tqdm progress bar parsing and native UI rendering.
    """
    finished_signal = pyqtSignal(int, str)  # exit_code, full_log

class ProcessConsoleWindow(QDialog):
    """
    A reusable dialog that executes an external command in a PyQt6 QProcess,
    displaying colored ANSI and rich console logs with clickable links (URLs/files).
    
    Supports optional real-time tqdm progress bar parsing and native UI rendering.
    """
    finished_signal = pyqtSignal(int, str)  # exit_code, full_log

    def __init__(self, command: str, working_dir: str = None, title: str = "Process Console", 
                 parent=None, on_finished=None, show_progress: bool = False):
        super().__init__(parent)
        self.command_str = command
        self.working_dir = working_dir or os.getcwd()
        self.on_finished_callback = on_finished
        self.show_progress = show_progress
        self.raw_ansi_log = ""
        self.line_buffer = ""
        
        self.ansi_parser = AnsiHtmlParser()
        
        # Idle tracking state
        self.last_output_time = time.time()
        self.current_idle_minutes = 0
        
        self.base_title = title
        self.update_status_title("Initializing")
        self.resize(800, 560)
        
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        app_inst = QApplication.instance()
        if app_inst:
            app_inst.aboutToQuit.connect(self.cleanup)
            
        self.setup_ui()
        
        # Setup QProcess
        self.process = QProcess(self)
        self.process.setWorkingDirectory(self.working_dir)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        
        # Configure process environment to force unbuffered output and UTF-8 encoding for Python subprocesses
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "UTF-8")
        env.insert("PYTHONUTF8", "1")
        self.process.setProcessEnvironment(env)
        
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.process_finished)
        
        # Setup Idle Monitor Timer
        self.idle_timer = QTimer(self)
        self.idle_timer.setInterval(1000)  # check every second
        self.idle_timer.timeout.connect(self.check_idle)
        self.idle_timer.start()
        
        # Start the process
        self.start_process()

    def update_status_title(self, status: str):
        self.setWindowTitle(f"{self.base_title} [{status}]")

    def cleanup(self):
        try:
            self.idle_timer.stop()
        except Exception:
            pass
        try:
            if self.process.state() == QProcess.ProcessState.Running:
                self.process.terminate()
                if not self.process.waitForFinished(1000):
                    self.process.kill()
        except Exception:
            pass

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # Text browser for rendering rich text with clickable anchors
        self.text_edit = QTextBrowser()
        self.text_edit.setReadOnly(True)
        self.text_edit.setOpenLinks(False)
        self.text_edit.anchorClicked.connect(self.on_link_clicked)
        self.text_edit.setFont(QFont("Consolas", 10))
        layout.addWidget(self.text_edit)
        
        # Footer
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(10)
        
        self.btn_kill = QPushButton("Kill")
        self.btn_kill.clicked.connect(self.kill_process)
        footer_layout.addWidget(self.btn_kill)
        
        # Borderless widget container for progress bar components next to Kill button
        self.progress_container = QWidget()
        progress_layout = QHBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(10)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(180)
        
        self.lbl_progress_prefix = QLabel("")
        self.lbl_progress_prefix.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        
        self.lbl_progress_info = QLabel("0/0 [00:00, 0it/s]")
        self.lbl_progress_info.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        
        progress_layout.addWidget(self.lbl_progress_prefix)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.lbl_progress_info)
        
        footer_layout.addWidget(self.progress_container)
        self.progress_container.hide()
        
        footer_layout.addStretch()
        
        self.btn_close = QPushButton("Close")
        self.btn_close.setVisible(False)  # Hidden initially
        self.btn_close.clicked.connect(self.accept)
        footer_layout.addWidget(self.btn_close)
        
        layout.addLayout(footer_layout)

    def start_process(self):
        try:
            # On Windows, set posix=False to preserve backslashes in paths
            args = shlex.split(self.command_str, posix=(sys.platform != 'win32'))
            if not args:
                raise ValueError("Command string is empty")
            program = args[0]
            arguments = args[1:]
            
            self.update_status_title("Running")
            
            # Print initial run info
            self.append_text(f"$ {self.command_str}\n")
            self.append_text(f"[Working Directory: {self.working_dir}]\n\n")
            
            self.process.start(program, arguments)
        except Exception as e:
            self.append_text(f"\n[ERROR] Failed to start process: {str(e)}\n")
            self.update_status_title("Failed to start")
            self.btn_close.setVisible(True)
            self.btn_kill.setVisible(False)
            if self.on_finished_callback:
                self.on_finished_callback(-1, self.get_plain_log())
            self.finished_signal.emit(-1, self.get_plain_log())

    def read_output(self):
        data = self.process.readAllStandardOutput()
        if not data:
            return
            
        text = bytes(data).decode('utf-8', errors='replace')
        
        # Reset idle timer state
        self.last_output_time = time.time()
        self.current_idle_minutes = 0
        
        # Prepend buffered line remainder
        text = self.line_buffer + text
        self.line_buffer = ""
        
        # Split into lines by \n or \r
        matches = list(re.finditer(r'[\r\n]', text))
        if not matches:
            self.line_buffer = text
            return
            
        last_end = 0
        for m in matches:
            line = text[last_end:m.start()]
            separator = m.group(0)
            self.process_line(line, separator)
            last_end = m.end()
            
        self.line_buffer = text[last_end:]

    def process_line(self, line: str, separator: str):
        # 1. Clean line from ANSI escape sequences to match TQDM regex
        clean_line = re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', line)
        
        # 2. Check for TQDM progress pattern (prefix-inclusive via re.search)
        # Matches: (prefix) percentage%, bar, current/total, [elapsed<remaining, rate]
        tqdm_pattern = r'(.*?)\s*(\d+)%\|(.*?)\| \s*(\d+)/(\d+) \[(.*?)\]'
        m = re.search(tqdm_pattern, clean_line)
        
        if self.show_progress and m:
            prefix = m.group(1).strip()
            percent = int(m.group(2))
            current = m.group(4)
            total = m.group(5)
            stats = m.group(6)
            
            info_text = f"{current}/{total} [{stats}]"
            
            # Update native progress bar UI
            self.update_progress_bar(percent, prefix, info_text)
        else:
            # Hide the progress bar container when regular log lines are printed
            if self.show_progress and not self.progress_container.isHidden():
                self.progress_container.hide()
            # Append regular line with its separator (retaining logs intact)
            self.append_text(line + separator)

    def update_progress_bar(self, percent: int, prefix: str, info_text: str):
        if self.progress_container.isHidden():
            self.progress_container.show()
        self.progress_bar.setValue(percent)
        self.lbl_progress_prefix.setText(prefix)
        self.lbl_progress_info.setText(info_text)

    def append_text(self, text: str):
        scrollbar = self.text_edit.verticalScrollBar()
        is_at_bottom = scrollbar.value() >= (scrollbar.maximum() - 10)
        
        self.raw_ansi_log += text
        
        html_chunk = self.ansi_parser.parse(text, linkify_files=True, working_dir=self.working_dir)
        
        cursor = self.text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertHtml(html_chunk)
        
        if is_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def check_idle(self):
        if self.process.state() != QProcess.ProcessState.Running:
            return
            
        elapsed = time.time() - self.last_output_time
        minutes_elapsed = int(elapsed // 60)
        
        if minutes_elapsed > self.current_idle_minutes:
            self.current_idle_minutes = minutes_elapsed
            self.append_text(f"\n(idle for {self.current_idle_minutes} minute{'s' if self.current_idle_minutes != 1 else ''})\n")

    def kill_process(self):
        if self.process.state() == QProcess.ProcessState.Running:
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()

    def process_finished(self, exit_code: int, exit_status):
        self.idle_timer.stop()
        
        # Flush line buffer
        if self.line_buffer:
            self.process_line(self.line_buffer, "")
            self.line_buffer = ""
            
        # Hide the progress bar container on finish
        if self.show_progress and not self.progress_container.isHidden():
            self.progress_container.hide()
                
        if exit_status == QProcess.ExitStatus.CrashExit:
            self.update_status_title("Crashed")
            self.append_text(f"\n[PROCESS CRASHED] Exit code: {exit_code}\n")
        elif exit_code != 0:
            self.update_status_title(f"Failed (Code {exit_code})")
            self.append_text(f"\n[PROCESS FAILED] Exit code: {exit_code}\n")
        else:
            self.update_status_title("Completed")
            self.append_text(f"\n[PROCESS COMPLETED SUCCESSFULLY]\n")
            
        self.btn_close.setVisible(True)
        self.btn_kill.setVisible(False)
        
        if self.on_finished_callback:
            try:
                self.on_finished_callback(exit_code, self.get_plain_log())
            except Exception as e:
                print(f"[ProcessConsoleWindow] Error in on_finished callback: {e}")
                
        self.finished_signal.emit(exit_code, self.get_plain_log())

    def on_link_clicked(self, qurl):
        if qurl.scheme() == "revealfile":
            path = qurl.path()
            if sys.platform == "win32" and path.startswith("/") and len(path) > 2 and path[2] == ":":
                path = path[1:]
            self.reveal_in_file_browser(path)
        else:
            import webbrowser
            webbrowser.open(qurl.toString())

    def reveal_in_file_browser(self, path: str):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return
            
        import subprocess
        
        # Try dbus-send on Linux to select the file
        if sys.platform.startswith("linux"):
            try:
                from urllib.request import pathname2url
                file_url = f"file://{pathname2url(path)}"
                cmd = [
                    "dbus-send", "--session", "--print-reply",
                    "--dest=org.freedesktop.FileManager1",
                    "/org/freedesktop/FileManager1",
                    "org.freedesktop.FileManager1.ShowItems",
                    f"array:string:{file_url}", "string:"
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
                return
            except Exception:
                pass
                
        # Fallback to standard platform commands
        try:
            parent_dir = os.path.dirname(path)
            if os.path.isdir(parent_dir):
                if sys.platform == "win32":
                    subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-R", path])
                else:
                    subprocess.Popen(["xdg-open", parent_dir])
        except Exception as e:
            print(f"[ProcessConsoleWindow] Error revealing path: {e}")

    def get_plain_log(self) -> str:
        # Strip ANSI escape sequences
        return re.sub(r'\x1b\[[0-9;?]*[a-zA-Z]', '', self.raw_ansi_log)

    def get_ansi_log(self) -> str:
        return self.raw_ansi_log



    def reject(self):
        self.close()

    def closeEvent(self, event):
        if self.process.state() == QProcess.ProcessState.Running:
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Terminate Process?",
                "The process is still running. Do you want to terminate it and close the window?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.cleanup()
                event.accept()
            else:
                event.ignore()
        else:
            self.cleanup()
            event.accept()


def run_process_in_console(command: str, working_dir: str = None, title: str = "Process Console", 
                           parent=None, on_finished=None, show_progress: bool = False) -> ProcessConsoleWindow:
    win = ProcessConsoleWindow(
        command=command,
        working_dir=working_dir,
        title=title,
        parent=parent,
        on_finished=on_finished,
        show_progress=show_progress
    )
    win.show()
    return win


def run_process_in_console_modal(command: str, working_dir: str = None, title: str = "Process Console", 
                                 parent=None, show_progress: bool = False) -> tuple[int, str]:
    win = ProcessConsoleWindow(
        command=command,
        working_dir=working_dir,
        title=title,
        parent=parent,
        show_progress=show_progress
    )
    win.exec()
    return win.process.exitCode(), win.get_plain_log()
