"""
Scoring Results Page (Demo Mode)

This page displays demonstration scoring results with fictional candidate data.
The hardcoded demo data represents potential radar sites in the Nice/French Riviera area.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Import shared utilities from main app
sys.path.insert(0, str(Path(__file__).parent.parent))

from radar_coverage_app import (
    apply_theme,
    BG_APP, BG_PANEL, BG_CARD, BORDER_COLOR,
    TEXT_PRIMARY, TEXT_SECONDARY,
    ACCENT_PRIMARY, ACCENT_HOVER,
)

# Page configuration
st.set_page_config(
    page_title="Scoring Results - Radar Coverage",
    page_icon="🏆",
    layout="wide"
)

# Apply theme
apply_theme()


def export_ranked_csv(df: pd.DataFrame) -> str:
    """Export all ranked candidates to CSV format."""
    lines = ["rank,latitude,longitude,score"]
    for _, row in df.iterrows():
        lines.append(f"{int(row['rank'])},{row['latitude']:.6f},{row['longitude']:.6f},{row['score']:.2f}")
    return "\n".join(lines)


def export_ranked_kmz(df: pd.DataFrame) -> bytes:
    """Export all ranked candidates to KMZ format for Google Earth."""
    import xml.etree.ElementTree as ET
    from io import BytesIO
    import zipfile
    
    n_candidates = len(df)
    
    # Create KML structure
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")
    ET.SubElement(document, "name").text = f"Ranked Radar Site Candidates ({n_candidates} sites)"
    ET.SubElement(document, "description").text = "Best candidate locations ranked by visibility score"
    
    # Define styles
    # Style for top 3 candidates (gold)
    style_top3 = ET.SubElement(document, "Style", id="top3_style")
    icon_style_top3 = ET.SubElement(style_top3, "IconStyle")
    ET.SubElement(icon_style_top3, "color").text = "ff00d7ff"  # Gold (AABBGGRR)
    ET.SubElement(icon_style_top3, "scale").text = "1.3"
    icon = ET.SubElement(icon_style_top3, "Icon")
    ET.SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/paddle/ylw-stars.png"
    
    # Style for other candidates (green)
    style_other = ET.SubElement(document, "Style", id="candidate_style")
    icon_style_other = ET.SubElement(style_other, "IconStyle")
    ET.SubElement(icon_style_other, "color").text = "ff00ff00"  # Green
    ET.SubElement(icon_style_other, "scale").text = "1.0"
    icon2 = ET.SubElement(icon_style_other, "Icon")
    ET.SubElement(icon2, "href").text = "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png"
    
    # Create folder for candidates
    folder = ET.SubElement(document, "Folder")
    ET.SubElement(folder, "name").text = f"All Candidates ({n_candidates})"
    ET.SubElement(folder, "description").text = "Ranked by weighted visibility score (FL5-FL400)"
    
    # Add placemarks for each candidate
    for _, row in df.iterrows():
        placemark = ET.SubElement(folder, "Placemark")
        rank = int(row['rank'])
        score = row['score']
        
        ET.SubElement(placemark, "name").text = f"#{rank} – {score:.2f}%"
        
        # Description with details
        desc = f"Rank: #{rank}\n"
        desc += f"Score: {score:.2f}%\n"
        desc += f"Latitude: {row['latitude']:.6f}°N\n"
        desc += f"Longitude: {row['longitude']:.6f}°E\n"
        if 'elevation_m' in row:
            desc += f"Elevation: {row['elevation_m']:.1f}m\n"
        ET.SubElement(placemark, "description").text = desc
        
        # Use gold style for top 3, green for others
        if rank <= 3:
            ET.SubElement(placemark, "styleUrl").text = "#top3_style"
        else:
            ET.SubElement(placemark, "styleUrl").text = "#candidate_style"
        
        # Add point geometry
        point = ET.SubElement(placemark, "Point")
        coords = ET.SubElement(point, "coordinates")
        coords.text = f"{row['longitude']},{row['latitude']},0"
    
    # Convert to KMZ (zipped KML)
    kml_string = ET.tostring(kml, encoding='unicode', method='xml')
    kml_bytes = f'<?xml version="1.0" encoding="UTF-8"?>\n{kml_string}'.encode('utf-8')
    
    kmz_buffer = BytesIO()
    with zipfile.ZipFile(kmz_buffer, 'w', zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr('doc.kml', kml_bytes)
    
    return kmz_buffer.getvalue()


def get_demo_data() -> pd.DataFrame:
    """
    Create a hardcoded demo DataFrame with fictional candidate data.
    Data represents potential radar sites in the Nice/French Riviera area.
    """
    demo_data = pd.DataFrame({
        'rank': range(1, 21),
        'latitude': [
            43.654578, 43.763829, 43.712345, 43.698421, 43.745632,
            43.681234, 43.729876, 43.756123, 43.692345, 43.718765,
            43.674532, 43.741098, 43.687654, 43.703456, 43.759321,
            43.666789, 43.723456, 43.748901, 43.695678, 43.734567
        ],
        'longitude': [
            7.078679, 7.274624, 7.156789, 7.198432, 7.234567,
            7.112345, 7.189012, 7.256789, 7.134567, 7.212345,
            7.089012, 7.245678, 7.167890, 7.178901, 7.267890,
            7.101234, 7.201234, 7.289012, 7.145678, 7.223456
        ],
        'elevation_m': [
            450.2, 380.5, 520.1, 412.8, 365.3,
            478.9, 395.6, 342.1, 445.7, 401.2,
            498.4, 358.9, 432.6, 467.3, 325.8,
            512.4, 389.7, 301.5, 456.2, 378.4
        ],
        'score': [
            87.45, 85.23, 82.67, 80.12, 78.56,
            76.89, 75.34, 73.21, 71.45, 69.87,
            68.23, 66.78, 65.12, 63.45, 61.89,
            60.23, 58.67, 56.45, 54.12, 52.34
        ]
    })
    return demo_data


def main():
    st.title("🏆 Scoring Results")
    st.caption("Ranked candidate sites by radar visibility coverage")
    
    # Demo mode banner
    st.info("📊 **Demo Mode**: Displaying sample scoring results with fictional candidate data in the Nice/French Riviera area.")
    
    # Always use hardcoded demo data
    ranked_df = get_demo_data()
    n_candidates = len(ranked_df)
    
    # Summary metrics
    st.markdown("---")
    st.subheader("Summary Statistics")
    
    if not ranked_df.empty:
        best_score = ranked_df['score'].iloc[0]
        avg_score = ranked_df['score'].mean()
        worst_score = ranked_df['score'].iloc[-1]
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("🥇 Best Score", f"{best_score:.2f}%")
        with metric_cols[1]:
            st.metric("📊 Average Score", f"{avg_score:.2f}%")
        with metric_cols[2]:
            st.metric("📉 Worst Score", f"{worst_score:.2f}%")
        with metric_cols[3]:
            st.metric("📍 Total Ranked", f"{n_candidates:,}")
    
    # Ranked candidates table
    st.markdown("---")
    st.subheader("Ranked Candidates")
    st.caption("All candidates ranked by weighted visibility score (FL5-FL400, with FL5/10/20 weighted x2)")
    
    # Display table with formatting
    st.dataframe(
        ranked_df[['rank', 'latitude', 'longitude', 'elevation_m', 'score']].style.format({
            'latitude': '{:.6f}',
            'longitude': '{:.6f}',
            'elevation_m': '{:.1f}',
            'score': '{:.2f}%'
        }).background_gradient(
            subset=['score'],
            cmap='Greens',
            vmin=0,
            vmax=100
        ),
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", help="Candidate ranking by weighted score"),
            "latitude": st.column_config.NumberColumn("Latitude", help="Latitude in degrees"),
            "longitude": st.column_config.NumberColumn("Longitude", help="Longitude in degrees"),
            "elevation_m": st.column_config.NumberColumn("Elevation (m)", help="Ground elevation in meters"),
            "score": st.column_config.NumberColumn("Score %", help="Weighted visibility score across FL5-FL400")
        }
    )
    
    # Export section
    st.markdown("---")
    st.subheader("Export Results")
    
    export_cols = st.columns(2)
    
    with export_cols[0]:
        st.markdown("##### CSV Format")
        st.caption("Spreadsheet-compatible format with all candidate data")
        csv_data = export_ranked_csv(ranked_df)
        st.download_button(
            f"📥 Download All {n_candidates} Candidates (CSV)",
            csv_data,
            "ranked_candidates.csv",
            "text/csv",
            use_container_width=True,
            key="export_csv_btn"
        )
    
    with export_cols[1]:
        st.markdown("##### KMZ Format")
        st.caption("Google Earth format for geographic visualization")
        kmz_data = export_ranked_kmz(ranked_df)
        st.download_button(
            f"📥 Download All {n_candidates} Candidates (KMZ)",
            kmz_data,
            "ranked_candidates.kmz",
            "application/vnd.google-earth.kmz",
            use_container_width=True,
            key="export_kmz_btn"
        )
    
    # Additional info
    st.markdown("---")
    st.markdown("""
    <div style='background-color: rgba(76, 175, 80, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50;'>
        <strong>Scoring Method</strong><br>
        Candidates are scored using weighted visibility analysis across multiple flight levels (FL5-FL400).<br>
        Flight levels FL5, FL10, and FL20 are weighted 2x to prioritize low-altitude coverage.
    </div>
    """, unsafe_allow_html=True)
    
    # Demo mode note
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: rgba(255, 193, 7, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #FFC107;'>
        <strong>Note</strong><br>
        This page displays demonstration data. In a production environment, actual scoring results 
        would be computed based on the constraints configured in the Site Selection page.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
