"""
Data Source Tracker - Tracks where every piece of data comes from
Location: src/data/data_source_tracker.py
"""

from datetime import datetime
from typing import Dict, Optional
import json
from pathlib import Path

class DataSourceTracker:
    """Track data sources for transparency"""
    
    def __init__(self):
        self.sources = {}
        self.methodologies = {}
        self.last_updated = {}
        
    def register_data_point(self, 
                           metric_name: str,
                           value: any,
                           source: str,
                           method: str,
                           confidence: str,
                           calculation_details: Optional[Dict] = None):
        """Register a data point with its source"""
        
        self.sources[metric_name] = {
            'value': value,
            'source': source,
            'method': method,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'calculation_details': calculation_details or {}
        }
        
    def get_source_info(self, metric_name: str) -> Dict:
        """Get source information for a metric"""
        return self.sources.get(metric_name, {
            'source': 'Unknown',
            'method': 'Not tracked',
            'confidence': 'Low',
            'timestamp': 'Unknown'
        })
    
    def format_for_display(self, metric_name: str) -> str:
        """Format source info for display"""
        info = self.get_source_info(metric_name)
        
        display = f"""
**Data Source:**
- Source: {info['source']}
- Method: {info['method']}
- Confidence: {info['confidence']}
- Last Updated: {info.get('timestamp', 'Unknown')}
        """
        
        if info.get('calculation_details'):
            display += "\n\n**Calculation Details:**\n"
            for key, value in info['calculation_details'].items():
                display += f"- {key}: {value}\n"
        
        return display.strip()
    
    def export_all_sources(self) -> Dict:
        """Export all tracked sources"""
        return {
            'sources': self.sources,
            'export_timestamp': datetime.now().isoformat()
        }
    
    def save_to_file(self, filepath: str):
        """Save sources to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.export_all_sources(), f, indent=2)