import pdfplumber
from pathlib import Path
import re
from typing import Optional, List

class DataPreprocessor:
    """Class to handle PDF processing and text cleaning for financial documents"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
        
    def clean_financial_text(self, text: str) -> str:
        """Clean financial document text by removing headers, footers, page numbers"""
        if not text:
            return ""
        
        # Remove specific TCS document headers and footers
        text = re.sub(r'This data can be easily copy pasted into a Microsoft Excel sheet PRINT', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Tata Consultancy Services Previous Years.*?�', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Source : Dion Global Solutions Limited\.?', '', text, flags=re.IGNORECASE)
        
        # Remove page numbers and markers
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Preserve line structure for financial data
        # Keep existing line breaks and don't collapse all whitespace
        text = re.sub(r'[ \t]+', ' ', text)  # Only collapse spaces/tabs, keep newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        # Clean up lines but preserve structure
        lines = []
        for line in text.split('\n'):
            cleaned_line = line.strip()
            if cleaned_line:  # Keep non-empty lines
                lines.append(cleaned_line)
            elif lines and lines[-1]:  # Add empty line only if previous line wasn't empty
                lines.append('')
        
        return '\n'.join(lines).strip()
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text and tables from PDF with proper formatting"""
        try:
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text_content.append(f"\n=== PAGE {page_num} ===\n")
                    
                    # Check if page has tables
                    tables = page.extract_tables()
                    
                    if tables:
                        # If tables exist, extract them with formatting
                        text_content.append(f"\n--- TABLES FROM PAGE {page_num} ---\n")
                        for table_num, table in enumerate(tables, 1):
                            text_content.append(f"\nTable {table_num}:\n")
                            text_content.append("-" * 50 + "\n")
                            
                            for row_num, row in enumerate(table):
                                if row:
                                    # Clean and format row
                                    cleaned_row = []
                                    for cell in row:
                                        cell_text = str(cell).strip() if cell else ""
                                        cleaned_row.append(cell_text)
                                    
                                    # Join cells with proper spacing for readability
                                    row_text = " | ".join(cleaned_row)
                                    text_content.append(row_text + "\n")
                                    
                                    # Add separator after header row
                                    if row_num == 0:
                                        text_content.append("-" * len(row_text) + "\n")
                            
                            text_content.append("\n")
                        
                        # Extract non-table text (filter out table content)
                        page_text = page.extract_text()
                        if page_text:
                            # Remove table-like content from regular text
                            non_table_text = self.filter_table_content(page_text)
                            if non_table_text.strip():
                                text_content.append(f"\n--- OTHER TEXT FROM PAGE {page_num} ---\n")
                                text_content.append(non_table_text)
                                text_content.append("\n")
                    else:
                        # No tables, extract regular text
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                            text_content.append("\n")
            
            return "".join(text_content)
        
        except Exception as e:
            print(f"Error extracting PDF: {str(e)}")
            return ""
    
    def filter_table_content(self, text: str) -> str:
        """Remove table-like content from regular text to avoid duplication"""
        if not text:
            return ""
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that look like table data (multiple numbers/values separated by spaces)
            if line and not re.match(r'^[\d\s\.\,\|\-]+$', line):
                # Skip lines with too many numeric patterns (likely table rows)
                numeric_parts = re.findall(r'\d+', line)
                if len(numeric_parts) < 4:  # Keep lines with fewer than 4 numbers
                    filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def segment_financial_statements(self, text: str) -> dict:
        """Segment financial data into logical sections"""
        segments = {
            'income_statement': [],
            'balance_sheet': [],
            'cash_flow': [],
            'other': []
        }
        
        current_section = 'other'
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Identify section based on keywords
            if any(keyword in line_lower for keyword in [
                'profit & loss', 'profit and loss', 'income statement', 
                'sales turnover', 'net sales', 'total income', 'operating profit',
                'reported net profit', 'earnings per share'
            ]):
                current_section = 'income_statement'
            
            elif any(keyword in line_lower for keyword in [
                'balance sheet', 'sources of funds', 'application of funds',
                'total share capital', 'networth', 'total assets', 'total liabilities',
                'current assets', 'current liabilities'
            ]):
                current_section = 'balance_sheet'
            
            elif any(keyword in line_lower for keyword in [
                'cash flow', 'net cash from operating', 'investing activities',
                'financing activities', 'cash and cash equivalents'
            ]):
                current_section = 'cash_flow'
            
            # Add line to current section
            if line.strip():
                segments[current_section].append(line)
        
        return segments
    
    def remove_duplicate_content(self, segments: dict) -> dict:
        """Remove duplicate tables and malformed content"""
        cleaned_segments = {}
        
        for section_name, content in segments.items():
            if not content:
                cleaned_segments[section_name] = []
                continue
            
            # Join content and split into logical blocks
            full_content = '\n'.join(content)
            
            # Remove duplicate table headers and malformed tables
            lines = full_content.split('\n')
            seen_lines = set()
            cleaned_lines = []
            
            for line in lines:
                line_cleaned = re.sub(r'\s+', ' ', line.strip())
                
                # Skip empty lines and page markers
                if not line_cleaned or 'PAGE' in line or '===' in line:
                    continue
                
                # Skip duplicate table markers
                if line_cleaned.startswith('Table') and line_cleaned in seen_lines:
                    continue
                
                # Skip malformed table content (too many separators)
                if line.count('|') > 10:
                    continue
                
                # Add unique meaningful content
                if line_cleaned not in seen_lines and len(line_cleaned) > 3:
                    seen_lines.add(line_cleaned)
                    cleaned_lines.append(line.strip())
            
            cleaned_segments[section_name] = cleaned_lines
        
        return cleaned_segments
    
    def format_segmented_report(self, segments: dict) -> str:
        """Format segmented financial data into readable report"""
        formatted_report = []
        
        # Income Statement Section
        if segments['income_statement']:
            formatted_report.append("=" * 60)
            formatted_report.append("INCOME STATEMENT & PROFIT & LOSS")
            formatted_report.append("=" * 60)
            formatted_report.extend(segments['income_statement'])
            formatted_report.append("\n")
        
        # Balance Sheet Section
        if segments['balance_sheet']:
            formatted_report.append("=" * 60)
            formatted_report.append("BALANCE SHEET")
            formatted_report.append("=" * 60)
            formatted_report.extend(segments['balance_sheet'])
            formatted_report.append("\n")
        
        # Cash Flow Section
        if segments['cash_flow']:
            formatted_report.append("=" * 60)
            formatted_report.append("CASH FLOW STATEMENT")
            formatted_report.append("=" * 60)
            formatted_report.extend(segments['cash_flow'])
            formatted_report.append("\n")
        
        # Other Information
        if segments['other']:
            formatted_report.append("=" * 60)
            formatted_report.append("OTHER INFORMATION")
            formatted_report.append("=" * 60)
            formatted_report.extend(segments['other'])
        
        return '\n'.join(formatted_report)
    
    def process_document(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Process document and return clean text"""
        
        if not Path(file_path).exists():
            print(f"File not found: {file_path}")
            return ""
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.supported_formats:
            print(f"Unsupported file format: {file_extension}")
            return ""
        
        print(f"Processing {Path(file_path).name}...")
        
        # Extract raw text
        if file_extension == '.pdf':
            raw_text = self.extract_pdf_text(file_path)
        else:
            return ""
        
        if not raw_text:
            print("Failed to extract text")
            return ""
        
        # Clean the text
        cleaned_text = self.clean_financial_text(raw_text)
        
        # Segment into financial statements
        print("Segmenting financial statements...")
        segments = self.segment_financial_statements(cleaned_text)
        
        # Remove duplicates and clean content
        print("Removing duplicates and cleaning content...")
        cleaned_segments = self.remove_duplicate_content(segments)
        
        # Format into structured report
        print("Formatting segmented report...")
        final_report = self.format_segmented_report(cleaned_segments)
        
        # Save individual segments only (no consolidated report)
        if output_path:
            base_path = Path(output_path).parent
            base_name = Path(output_path).stem
            
            segment_files = []
            for section_name, content in cleaned_segments.items():
                if content:
                    section_file = base_path / f"{base_name}_{section_name}.txt"
                    section_text = '\n'.join(content)
                    self.save_text(section_text, str(section_file))
                    segment_files.append(str(section_file))
            
            print(f"Individual segment files created: {len(segment_files)} files")
            for file in segment_files:
                print(f"• {Path(file).name}")
        
        print(f"Document processed and segmented successfully!")
        return final_report  # Return for display purposes only
    
    def save_text(self, text: str, output_path: str) -> bool:
        """Save text to file"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Text saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Failed to save: {str(e)}")
            return False
    
    def get_document_info(self, file_path: str) -> dict:
        """Get basic information about the document"""
        try:
            path = Path(file_path)
            info = {
                "filename": path.name,
                "file_size": path.stat().st_size,
                "format": path.suffix.lower()
            }
            
            if path.suffix.lower() == '.pdf':
                with pdfplumber.open(file_path) as pdf:
                    info["total_pages"] = len(pdf.pages)
            
            return info
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    print("Starting TCS Financial Statement Processing...")
    
    # File paths (relative to project root)
    input_pdf = "data/financial-statements/tcs financial statement.pdf"
    output_path = "data/financial-statements/tcs_clean.txt"
    
    print(f"Input: {input_pdf}")
    print(f"Output: {output_path}")
    print("-" * 50)
    
    # Initialize and run preprocessor
    preprocessor = DataPreprocessor()
    result = preprocessor.process_document(input_pdf, output_path)
    
    print("-" * 50)
    if result:
        print("Processing completed successfully!")
    else:
        print("Processing failed!")