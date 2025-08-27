from fpdf import FPDF
from typing import Dict, Any, List, Optional
import logging
import re

# PDF class methods
class PDF(FPDF):
    def __init__(self, theme="light", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theme = theme
        self.colors = self._get_theme_colors(theme)
        
    def _get_theme_colors(self, theme: str) -> Dict[str, tuple]:
        """Get color scheme based on theme."""
        themes = {
            "light": {
                "background": (255, 255, 255),  # White
                "text": (0, 0, 0),              # Black
                "header": (41, 128, 185),        # Blue
                "divider": (149, 165, 166),      # Light gray
                "accent": (52, 152, 219),        # Light blue
                "section_bg": (248, 249, 250)    # Very light gray
            },
            "dark": {
                "background": (33, 37, 41),      # Dark gray
                "text": (248, 249, 250),         # Light gray
                "header": (52, 152, 219),        # Blue
                "divider": (108, 117, 125),      # Medium gray
                "accent": (255, 193, 7),         # Yellow accent
                "section_bg": (52, 58, 64)       # Slightly lighter dark
            }
        }
        return themes.get(theme, themes["light"])

    def add_page(self, *args, **kwargs):
        super().add_page(*args, **kwargs)
        # Set background color for dark theme
        if self.theme == "dark":
            self.set_fill_color(*self.colors["background"])
            self.rect(0, 0, self.w, self.h, 'F')
    
    def header(self):
        # Only show the title on the first page
        if self.page_no() == 1:
            # Set header color
            self.set_text_color(*self.colors["header"])
            self.set_font('Arial', 'B', 14)
            
            title = 'Enhanced NLP Marketing Analysis Report'
            if self.theme == "dark":
                title += ' (Dark Mode)'
                
            self.cell(0, 10, title, 0, 1, 'C')
            self.ln(5)
            
            # Add a line under the header with theme color
            self.set_draw_color(*self.colors["divider"])
            self.set_line_width(0.5)
            self.line(10, 25, 200, 25)
            self.ln(10)
        else:
            # For subsequent pages, just add some top margin
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_text_color(*self.colors["text"])
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title, size=12):
        # Section background for dark theme
        if self.theme == "dark":
            self.set_fill_color(*self.colors["section_bg"])
            self.rect(10, self.get_y(), 190, 8, 'F')
            
        self.set_text_color(*self.colors["accent"])
        self.set_font('Arial', 'B', size)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(3)

    def chapter_body(self, body, font_size=11):
        self.set_text_color(*self.colors["text"])
        self.set_font('Arial', '', font_size)
        # Clean text to ensure ASCII compatibility
        cleaned_body = self._clean_text_for_pdf(body)
        self.multi_cell(0, 6, cleaned_body)
        self.ln(5)
        
    def _clean_text_for_pdf(self, text):
        """Clean text to ensure PDF compatibility."""
        if not text:
            return ""
        
        # Replace problematic Unicode characters with ASCII equivalents
        replacements = {
            '•': '-',  # bullet point
            '–': '-',  # en dash
            '—': '--', # em dash
            ''': "'",  # left single quote
            ''': "'",  # right single quote
            '"': '"',  # left double quote
            '"': '"',  # right double quote
            '…': '...', # ellipsis
        }
        
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        
        # Remove any remaining non-ASCII characters
        try:
            text.encode('latin-1')
            return text
        except UnicodeEncodeError:
            # Fallback: keep only ASCII characters
            return ''.join(char for char in text if ord(char) < 128)
    
    def _clean_orphaned_markdown(self, content: str) -> str:
        """Clean up orphaned markdown characters that appear on their own lines."""
        import re
        
        # Split into lines
        lines = content.split('\n')
        cleaned_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            
            # Skip completely empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                i += 1
                continue
            
            # Check if this line is just orphaned markdown characters
            if re.match(r'^[\*_`~#+\-=]+\s*$', stripped_line):
                # Skip lines that are just markdown characters (**, *, ***, __, _, etc.)
                i += 1
                continue
            
            # Check for lines that are just ** with optional whitespace
            if re.match(r'^\*\*\s*$', stripped_line):
                # Skip lines with just ** and whitespace
                i += 1
                continue
                
            # Check for lines that have only ** and maybe a few characters
            if re.match(r'^\*\*\s*[\w\s]{0,3}\s*$', stripped_line):
                # Skip lines like "** ", "**a", "** x", etc.
                i += 1
                continue
            
            # CRITICAL FIX: Check if the previous line was a VC question heading
            # and this line is just orphaned ** markers
            if i > 0 and stripped_line == '**':
                prev_line = lines[i-1].strip()
                # Check if previous line looks like a VC question heading
                if (re.match(r'^\*?\*?\d+\.\s', prev_line) and 
                    (':' in prev_line or any(keyword in prev_line for keyword in 
                        ['Assessment', 'Progress', 'Traction', 'Problem', 'Solution', 
                         'Market', 'Product', 'Technology', 'Monetization', 'Team',
                         'Competitive', 'Funding', 'Capital', 'Business', 'Strategy']))):
                    # Skip this orphaned ** line completely
                    i += 1
                    continue
            
            # Clean up lines that start with orphaned ** followed by content
            if stripped_line.startswith('**') and not stripped_line.endswith('**'):
                # Count ** occurrences
                asterisk_count = stripped_line.count('**')
                if asterisk_count == 1:
                    # Single ** at start - remove it
                    cleaned_line = stripped_line[2:].strip()
                    if cleaned_line:  # Only add if there's meaningful content left
                        cleaned_lines.append(cleaned_line)
                    i += 1
                    continue
                
            # Clean up lines that end with orphaned **
            if stripped_line.endswith('**') and not stripped_line.startswith('**'):
                asterisk_count = stripped_line.count('**')
                if asterisk_count == 1:
                    # Single ** at end - remove it
                    cleaned_line = stripped_line[:-2].strip()
                    if cleaned_line:  # Only add if there's meaningful content left
                        cleaned_lines.append(cleaned_line)
                    i += 1
                    continue
            
            # Keep the line as is
            cleaned_lines.append(line)
            i += 1
        
        return '\n'.join(cleaned_lines)
        
    def section_divider(self):
        """Add a visual divider between sections with theme colors."""
        self.set_draw_color(*self.colors["divider"])
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(h=10)
    
    def process_markdown_content(self, content: str):
        """Process markdown content and render it with proper PDF formatting."""
        if not content:
            return
        
        # Clean up orphaned markdown characters first
        content = self._clean_orphaned_markdown(content)
        
        # Split content into lines for processing
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines but add spacing
            if not line:
                self.ln(3)
                continue
            
            # Skip lines that are just orphaned markdown characters
            if line in ['**', '*', '***', '__', '_']:
                continue
                
            # Handle horizontal rule separator
            if line.startswith('---'):
                self.section_divider()
                continue
            
            # Handle headings
            if line.startswith('## '):
                # H2 heading - Main section
                heading_text = line[3:].replace('**', '')  # Remove markdown bold
                self.chapter_title(heading_text, 14)
                continue
            elif line.startswith('### '):
                # H3 heading - Subsection
                heading_text = line[4:].replace('**', '')  # Remove markdown bold
                self.chapter_title(heading_text, 12)
                continue
            elif line.startswith('# '):
                # H1 heading - Very large section
                heading_text = line[2:].replace('**', '')  # Remove markdown bold
                self.chapter_title(heading_text, 16)
                continue
            
            # Handle different bullet point formats
            if line.startswith('• ') or line.startswith('- ') or line.startswith('* '):
                # Determine bullet character and extract text
                bullet_text = ""  # Initialize to avoid unbound variable
                if line.startswith('• '):
                    bullet_text = line[2:]  # Remove bullet
                elif line.startswith('- '):
                    bullet_text = line[2:]  # Remove dash
                elif line.startswith('* '):
                    bullet_text = line[2:]  # Remove asterisk
                
                # Check for structured weakness format
                if '**Weakness:**' in bullet_text:
                    # Special formatting for weakness analysis
                    self._render_weakness_point(bullet_text)
                else:
                    # Regular bullet point with bold text support
                    self._render_bullet_point(bullet_text)
                continue
            
            # Handle numbered lists - but check for VC analysis question headings first
            import re
            if re.match(r'^\*?\*?\d+\. ', line):
                # This is a numbered line - check if it's a VC question heading
                # VC question headings typically have a colon and look like question titles
                
                # Extract the number and check if this looks like a question
                match = re.match(r'^\*?\*?(\d+)\. (.+)$', line)
                if match:
                    number = match.group(1)
                    rest_of_line = match.group(2)
                    
                    # IMPROVED: Better detection of VC question headings
                    # Check if this looks like a VC question heading
                    vc_keywords = ['Assessment', 'Progress', 'Traction', 'Problem', 'Solution', 
                                  'Market', 'Product', 'Technology', 'Monetization', 'Team',
                                  'Competitive', 'Funding', 'Capital', 'Business', 'Strategy',
                                  'Identification', 'Severity', 'Innovation', 'Differentiation']
                    
                    is_vc_question = (':' in rest_of_line or 
                                    any(keyword in rest_of_line for keyword in vc_keywords) or
                                    len(rest_of_line.split()) <= 6)  # Short titles are likely headings
                    
                    if is_vc_question:
                        # This is a VC question heading - render it properly
                        self._render_vc_question_heading(line)
                        continue
                    else:
                        # Regular numbered list item
                        self._render_numbered_item(number, rest_of_line)
                        continue
            
            # Handle paragraphs with bold text
            if '**' in line:
                self._render_paragraph_with_formatting(line)
            else:
                # Handle regular paragraphs
                cleaned_text = self._clean_text_for_pdf(line)
                self.set_text_color(*self.colors["text"])
                self.set_font('Arial', '', 11)
                self.multi_cell(0, 6, cleaned_text)
                self.ln(3)
    
    def _process_inline_formatting(self, text: str) -> str:
        """Process inline markdown formatting like bold text."""
        import re
        
        # Convert markdown links [text](url) to just text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        return text
    
    def _render_vc_question_heading(self, line: str):

    
    # Check if this line contains both heading and content separated by colon
        if ':' in line:
            colon_pos = line.find(':')
            heading_part = line[:colon_pos + 1]  # Include the colon
            content_part = line[colon_pos + 1:].strip()  # Content after colon
            
            # Extract and clean the heading text
            cleaned_heading = heading_part.replace('**', '').strip()
            
            # Set up for section heading - add space before
            self.ln(4)  # More space before the heading for better separation
            
            # Render the heading in accent color
            self.set_text_color(*self.colors["accent"])  
            self.set_font('Arial', 'B', 12)  # Bold, 12pt font for good visibility
            
            # Clean the heading text for PDF compatibility
            cleaned_heading_text = self._clean_text_for_pdf(cleaned_heading)
            
            # Render the heading
            self.cell(0, 8, cleaned_heading_text, 0, 1, 'L')
            self.ln(3)  # Space after heading before content starts
            
            # If there's content after the colon, render it as regular text
            if content_part:
                self.set_text_color(*self.colors["text"])
                self.set_font('Arial', '', 11)
                
                # Check if content has bold formatting
                if '**' in content_part:
                    self._render_text_with_bold(content_part)
                else:
                    cleaned_content = self._clean_text_for_pdf(content_part)
                    self.multi_cell(0, 6, cleaned_content)
                    self.ln(3)
        else:
            # No colon found - treat entire line as heading
            cleaned_line = line.replace('**', '').strip()
            
            # Ensure it ends with a colon for consistency
            if not cleaned_line.endswith(':'):
                cleaned_line += ':'
            
            # Set up for section heading - add space before
            self.ln(4)  # More space before the heading for better separation
            
            # Use accent color for headings in both light and dark themes
            self.set_text_color(*self.colors["accent"])  
            self.set_font('Arial', 'B', 12)  # Bold, 12pt font for good visibility
            
            # Clean the text for PDF compatibility
            cleaned_text = self._clean_text_for_pdf(cleaned_line)
            
            # Render as a section heading (no indentation, left-aligned)
            self.cell(0, 8, cleaned_text, 0, 1, 'L')
            self.ln(3)  # Space after heading before content starts
        
    def _render_bullet_point(self, text: str):
        """Render a bullet point with proper formatting and bold text support."""
        self.cell(5)  # Indent
        self.set_text_color(*self.colors["text"])
        
        # Check if text contains bold formatting
        if '**' in text:
            self._render_text_with_bold(f"- {text}", indent=0)
        else:
            self.set_font('Arial', '', 11)
            cleaned_text = self._clean_text_for_pdf(f"- {text}")
            self.multi_cell(0, 6, cleaned_text)
        
        self.ln(2)
    
    def _render_numbered_item(self, number: str, text: str):
        """Render a numbered list item with proper formatting."""
        self.cell(5)  # Indent
        self.set_text_color(*self.colors["text"])
        
        # Check if text contains bold formatting
        if '**' in text:
            self._render_text_with_bold(f"{number}. {text}", indent=0)
        else:
            self.set_font('Arial', '', 11)
            cleaned_text = self._clean_text_for_pdf(f"{number}. {text}")
            self.multi_cell(0, 6, cleaned_text)
        
        self.ln(2)
    
    def _render_paragraph_with_formatting(self, text: str):
        """Render a paragraph that may contain bold text."""
        self.set_text_color(*self.colors["text"])
        self._render_text_with_bold(text)
        self.ln(3)
    
    def _render_text_with_bold(self, text: str, indent: int = 0):
        """Render text with bold formatting support."""
        import re
        
        if indent > 0:
            self.cell(indent)
        
        # Split text by bold markers
        parts = re.split(r'(\*\*.*?\*\*)', text)
        
        current_x = self.get_x()
        current_y = self.get_y()
        line_height = 6
        
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                # Bold text
                bold_text = part[2:-2]  # Remove ** markers
                cleaned_bold = self._clean_text_for_pdf(bold_text)
                
                self.set_font('Arial', 'B', 11)
                
                # Check if text fits on current line
                text_width = self.get_string_width(cleaned_bold)
                if current_x + text_width > 200:  # Near right margin
                    self.ln(line_height)
                    current_x = 10
                    current_y = self.get_y()
                
                self.set_xy(current_x, current_y)
                self.cell(text_width, line_height, cleaned_bold, 0, 0)
                current_x += text_width
                
            else:
                # Regular text
                if part.strip():  # Only process non-empty parts
                    cleaned_text = self._clean_text_for_pdf(part)
                    
                    self.set_font('Arial', '', 11)
                    
                    # Handle line wrapping for long text
                    words = cleaned_text.split(' ')
                    for i, word in enumerate(words):
                        if i > 0:
                            word = ' ' + word
                        
                        word_width = self.get_string_width(word)
                        if current_x + word_width > 200:  # Near right margin
                            self.ln(line_height)
                            current_x = 10
                            current_y = self.get_y()
                            word = word.lstrip()  # Remove leading space after line break
                            word_width = self.get_string_width(word)
                        
                        self.set_xy(current_x, current_y)
                        self.cell(word_width, line_height, word, 0, 0)
                        current_x += word_width
        
        # Move to next line after rendering all parts
        self.ln(line_height)
    
    def _render_weakness_point(self, text: str):
        """Render a structured weakness analysis point with proper formatting."""
        # Split the weakness point into components
        if '**Weakness:**' in text and '**Why it Fails:**' in text and '**Actionable Recommendation:**' in text:
            # Extract components using more robust parsing
            parts = text.split('**Why it Fails:**')
            if len(parts) >= 2:
                weakness_part = parts[0].replace('**Weakness:**', '').strip()
                
                remaining = parts[1].split('**Actionable Recommendation:**')
                if len(remaining) >= 2:
                    why_part = remaining[0].strip()
                    rec_part = remaining[1].strip()
                    
                    # Render with proper formatting and indentation
                    self.cell(5)  # Indent
                    self.set_text_color(*self.colors["accent"])
                    self.set_font('Arial', 'B', 11)
                    self.cell(0, 6, "Weakness:", 0, 1)
                    
                    self.cell(10)  # More indent
                    self.set_text_color(*self.colors["text"])
                    self.set_font('Arial', '', 10)
                    cleaned_weakness = self._clean_text_for_pdf(weakness_part)
                    self.multi_cell(0, 5, cleaned_weakness)
                    
                    self.cell(5)  # Indent
                    self.set_text_color(*self.colors["accent"])
                    self.set_font('Arial', 'B', 11)
                    self.cell(0, 6, "Why it Fails:", 0, 1)
                    
                    self.cell(10)  # More indent
                    self.set_text_color(*self.colors["text"])
                    self.set_font('Arial', '', 10)
                    cleaned_why = self._clean_text_for_pdf(why_part)
                    self.multi_cell(0, 5, cleaned_why)
                    
                    self.cell(5)  # Indent
                    self.set_text_color(*self.colors["accent"])
                    self.set_font('Arial', 'B', 11)
                    self.cell(0, 6, "Actionable Recommendation:", 0, 1)
                    
                    self.cell(10)  # More indent
                    self.set_text_color(*self.colors["text"])
                    self.set_font('Arial', '', 10)
                    cleaned_rec = self._clean_text_for_pdf(rec_part)
                    self.multi_cell(0, 5, cleaned_rec)
                    
                    self.ln(4)  # Extra space after weakness point
                    return
        
        # Fallback: render as regular bullet point with bold support
        self._render_bullet_point(text)

def create_enhanced_report_pdf(
    original_text: str,
    enhanced_report: Dict[str, Any],
    output_filename: str = "Enhanced_NLP_Report.pdf",
    theme: str = "light"
) -> None:
    """
    Generates an enhanced PDF report with structured marketing analysis.

    Args:
        original_text: The full source text.
        enhanced_report: The enhanced analysis report dictionary.
        output_filename: The name of the output PDF file.
        theme: PDF theme - "light" or "dark"
    """
    pdf = PDF(theme=theme)
    pdf.add_page()

    # 1. Executive Summary Section
    pdf.chapter_title("1. Executive Summary", 14)
    executive_summary = enhanced_report.get('executive_summary', 'No executive summary available.')
    pdf.chapter_body(executive_summary, 12)
    pdf.section_divider()

    # 2. Marketing Goal Section
    pdf.chapter_title("2. Marketing Goal", 14)
    marketing_goal = enhanced_report.get('marketing_goal', 'Not identified')
    pdf.chapter_body(f"Primary Marketing Structure: {marketing_goal}", 12)
    pdf.section_divider()

    # 3. Key Entities & Roles Section
    pdf.chapter_title("3. Key Entities & Roles", 14)
    entities_roles = enhanced_report.get('key_entities_roles', [])
    
    if entities_roles:
        entities_text = ""
        for entity_info in entities_roles:
            entity = entity_info.get('entity', 'Unknown')
            entity_type = entity_info.get('type', 'N/A')
            role = entity_info.get('role', 'No role defined')
            # Use ASCII-compatible bullet point
            entities_text += f"- {entity} ({entity_type}): {role}\n"
    else:
        entities_text = "No entities identified in the text."
    
    pdf.chapter_body(entities_text, 11)
    pdf.section_divider()

    # 4. Persuasive Keywords Section
    pdf.chapter_title("4. Persuasive Keywords", 14)
    keywords = enhanced_report.get('persuasive_keywords', [])
    
    if keywords:
        keywords_text = "Key marketing words identified: " + ", ".join(keywords)
    else:
        keywords_text = "No specific persuasive keywords identified."
    
    pdf.chapter_body(keywords_text, 11)
    pdf.section_divider()

    # 5. Additional Analysis (Optional)
    metadata = enhanced_report.get('analysis_metadata', {})
    if metadata:
        pdf.chapter_title("5. Analysis Details", 12)
        details_text = f"Text Length: {metadata.get('text_length', 'N/A')} characters\n"
        details_text += f"Entities Found: {metadata.get('entity_count', 'N/A')}\n"
        details_text += f"Processing Notes: {metadata.get('processing_notes', 'Standard analysis')}"
        pdf.chapter_body(details_text, 10)
        pdf.section_divider()

    # 6. Full Source Text Section
    pdf.chapter_title("6. Full Source Text", 12)
    pdf.chapter_body(original_text, 10)

    pdf.output(output_filename)

def create_report_pdf(
    original_text: str,
    summary: str,
    entities: list[dict],
    output_filename: str = "NLP_Report.pdf",
    theme: str = "light"
) -> None:
    """
    Generates a basic PDF report with the NLP analysis results (for backward compatibility).

    Args:
        original_text: The full source text.
        summary: The generated summary.
        entities: A list of extracted named entities.
        output_filename: The name of the output PDF file.
        theme: PDF theme - "light" or "dark"
    """
    pdf = PDF(theme=theme)
    pdf.add_page()

    # Summary Section
    pdf.chapter_title("Executive Summary")
    pdf.chapter_body(summary)

    # Entities Section
    pdf.chapter_title("Extracted Entities")
    entity_text = "\n".join([f"- {entity['word']} ({entity['entity_group']})" for entity in entities])
    if not entity_text:
        entity_text = "No entities found."
    pdf.chapter_body(entity_text)

    # Original Text Section
    pdf.chapter_title("Full Source Text")
    pdf.chapter_body(original_text)

    pdf.output(output_filename)

def create_vc_analysis_pdf(
    pitch_text: str,
    vc_analysis: str,
    output_filename: str = "VC_Pitch_Analysis.pdf",
    theme: str = "light",
    include_original_pitch: bool = False  # Changed default to False
) -> None:
    """
    Generates a comprehensive VC pitch analysis PDF report with structured analysis.
    
    Args:
        pitch_text: The original business pitch text
        vc_analysis: The VC analysis results as a formatted string
        output_filename: The name of the output PDF file
        theme: PDF theme - "light" or "dark"
        include_original_pitch: Whether to include the original pitch text (now defaults to False)
    """
    import logging
    
    logging.info(f"📄 Starting PDF creation: {output_filename}")
    logging.info(f"Theme: {theme}, Include original pitch: {include_original_pitch}")
    logging.info(f"Analysis content length: {len(vc_analysis)} characters")
    
    pdf = PDF(theme=theme)
    pdf.add_page()
    logging.info("✅ PDF instance created and page added")
    
    # Title and Executive Summary
    pdf.chapter_title("VC Pitch Analysis Report", 16)
    logging.info("✅ PDF title added")
    
    # Analysis Content - Process as markdown for proper formatting
    if vc_analysis and vc_analysis.strip():
        logging.info("🔄 Processing analysis content as markdown...")
        pdf.process_markdown_content(vc_analysis)
        logging.info("✅ Analysis content processed and added to PDF")
    else:
        logging.warning("⚠️ No analysis content available")
        pdf.chapter_body("No analysis available", 11)
    
    pdf.section_divider()
    
    # Include original pitch only if explicitly requested
    if include_original_pitch:
        logging.info("📃 Adding original pitch text to PDF...")
        pdf.chapter_title("Original Pitch Text (For Reference)", 12)
        pdf.chapter_body(pitch_text, 9)
        logging.info("✅ Original pitch text added to PDF")
    
    logging.info(f"💾 Saving PDF to: {output_filename}")
    pdf.output(output_filename)
    
    # Verify file was created
    import os
    if os.path.exists(output_filename):
        file_size = os.path.getsize(output_filename)
        logging.info(f"✅ PDF created successfully: {output_filename} ({file_size} bytes)")
    else:
        logging.error(f"❌ PDF file was not created: {output_filename}")

def create_comprehensive_report_pdf(
    original_text: str,
    enhanced_report: Optional[Dict[str, Any]] = None,
    vc_analysis: Optional[Dict[str, Any]] = None,
    output_filename: str = "Comprehensive_Analysis_Report.pdf",
    theme: str = "light"
) -> None:
    """
    Generates a comprehensive PDF report combining enhanced NLP analysis and VC pitch analysis.
    
    Args:
        original_text: The full source text
        enhanced_report: The enhanced NLP analysis report (optional)
        vc_analysis: The VC pitch analysis results (optional) 
        output_filename: The name of the output PDF file
        theme: PDF theme - "light" or "dark"
    """
    pdf = PDF(theme=theme)
    pdf.add_page()
    
    # Main Title
    pdf.chapter_title("Comprehensive Business Analysis Report", 16)
    pdf.ln(5)
    
    # Table of Contents
    pdf.chapter_title("Table of Contents", 12)
    toc_items = []
    if enhanced_report:
        toc_items.extend([
            "1. Executive Summary",
            "2. Marketing Goal Analysis", 
            "3. Key Entities & Roles",
            "4. Persuasive Keywords"
        ])
    if vc_analysis:
        toc_items.extend([
            "5. Venture Capital Analysis",
            "6. Investment Readiness Assessment"
        ])
    toc_items.append("7. Full Source Text")
    
    toc_text = "\n".join([f"{item}" for item in toc_items])
    pdf.chapter_body(toc_text, 10)
    pdf.section_divider()
    
    # Enhanced NLP Report Section (if available)
    if enhanced_report:
        pdf.chapter_title("1. Executive Summary", 14)
        executive_summary = enhanced_report.get('executive_summary', 'No executive summary available.')
        pdf.chapter_body(executive_summary, 12)
        pdf.section_divider()
        
        pdf.chapter_title("2. Marketing Goal Analysis", 14)
        marketing_goal = enhanced_report.get('marketing_goal', 'Not identified')
        pdf.chapter_body(f"Primary Marketing Structure: {marketing_goal}", 12)
        pdf.section_divider()
        
        pdf.chapter_title("3. Key Entities & Roles", 14)
        entities_roles = enhanced_report.get('key_entities_roles', [])
        
        if entities_roles:
            entities_text = ""
            for entity_info in entities_roles:
                entity = entity_info.get('entity', 'Unknown')
                entity_type = entity_info.get('type', 'N/A')
                role = entity_info.get('role', 'No role defined')
                entities_text += f"- {entity} ({entity_type}): {role}\n"
        else:
            entities_text = "No entities identified in the text."
        
        pdf.chapter_body(entities_text, 11)
        pdf.section_divider()
        
        pdf.chapter_title("4. Persuasive Keywords", 14)
        keywords = enhanced_report.get('persuasive_keywords', [])
        
        if keywords:
            keywords_text = "Key marketing words identified: " + ", ".join(keywords)
        else:
            keywords_text = "No specific persuasive keywords identified."
        
        pdf.chapter_body(keywords_text, 11)
        pdf.section_divider()
    
    # VC Analysis Section (if available)
    if vc_analysis and 'batch_analyses' in vc_analysis:
        pdf.chapter_title("5. Venture Capital Analysis", 14)
        
        # Analysis Configuration
        config = vc_analysis.get('analysis_config', {})
        config_text = f"Analysis performed using {config.get('num_runs', 'N/A')} LLM runs with "
        config_text += f"{config.get('questions_per_run', 'N/A')} questions per run distribution."
        pdf.chapter_body(config_text, 10)
        pdf.ln(3)
        
        # Batch Analyses Summary
        batch_analyses = vc_analysis.get('batch_analyses', [])
        for i, batch in enumerate(batch_analyses):
            pdf.chapter_title(f"Batch {batch.get('batch_number', i+1)} - {batch.get('questions_analyzed', 'N/A')} Questions", 12)
            analysis_text = batch.get('analysis', 'No analysis available')
            # Limit batch analysis length for comprehensive report
            if len(analysis_text) > 1000:
                analysis_text = analysis_text[:1000] + "... [Analysis truncated in comprehensive view]"
            pdf.chapter_body(analysis_text, 10)
            pdf.ln(3)
        
        pdf.section_divider()
        
        # Overall Investment Verdict with markdown processing
        pdf.chapter_title("6. Investment Readiness Assessment", 14)
        overall_verdict = vc_analysis.get('overall_verdict', 'No overall verdict available')
        
        # Use markdown processing for properly formatted output
        pdf.process_markdown_content(overall_verdict)
        pdf.section_divider()
    
    # Source Text
    pdf.chapter_title("7. Full Source Text", 12)
    pdf.chapter_body(original_text, 10)
    
    pdf.output(output_filename)