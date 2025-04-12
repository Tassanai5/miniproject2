import streamlit as st
import numpy as np
import pandas as pd
import time

# === CONFIGURATION ===
NV_ROOT = '/Users/Tassanai/Library/CloudStorage/OneDrive-ChulalongkornUniversity/University/3Junior/Second Semester/2603278 INFO VISUAL/miniproject 2/dataset/NV'
CLIP_VIDEO_MAPPING = {
    'Shrek': {"Low": ["SHREK_10a", "HEATMAPURL:https://res.cloudinary.com/dlggzzrag/video/upload/v1744443658/heatmapoverlay_SHREK_10a_s0qq7j.mp4"], "Medium": ["SHREK_3b", "HEATMAPURL:https://res.cloudinary.com/dlggzzrag/video/upload/v1744443668/heatmapoverlay_SHREK_3b_lqcyyc.mp4"], "High": ["SHREK_3a", "HEATMAPURL:https://res.cloudinary.com/dlggzzrag/video/upload/v1744443692/heatmapoverlay_SHREK_3a_dgrkux.mp4"], "ThumbnailURL": "https://github.com/Tassanai5/material-miniproject2/blob/main/image/Shrek_Thumbnail.jpg?raw=true"},
    'Deep Blue': {"Low": ["DEEPB_9c", "HEATMAPURL:https://res.cloudinary.com/dlggzzrag/video/upload/v1744443688/heatmapoverlay_DEEPB_9c_uwtbxs.mp4"], "Medium": ["DEEPB_5b", "HEATMAPURL:https://res.cloudinary.com/dlggzzrag/video/upload/v1744443488/heatmapoverlay_DEEPB_5b_p1erof.mp4"], "High": ["DEEPB_11a", "HEATMAPURL:https://res.cloudinary.com/dlggzzrag/video/upload/v1744443720/heatmapoverlay_DEEPB_11a_packhk.mp4"], "ThumbnailURL": "https://github.com/Tassanai5/material-miniproject2/blob/main/image/Deep_Blue_Thumbnail.jpg?raw=true"},
    'The Squid and the Whale': {"Low": ["SQUID_8a", "HEATMAPURL:https://res.cloudinary.com/dlggzzrag/video/upload/v1744443689/heatmapoverlay_SQUID_8a_vehssz.mp4"], "Medium": ["SQUID_12c", "HEATMAPURL:https://res.cloudinary.com/dlggzzrag/video/upload/v1744443695/heatmapoverlay_SQUID_12c_x7zaou.mp4"], "High": ["SQUID_6a", "HEATMAPURL:https://res.cloudinary.com/dlggzzrag/video/upload/v1744443691/heatmapoverlay_SQUID_6a_wpmqec.mp4"], "ThumbnailURL": "https://github.com/Tassanai5/material-miniproject2/blob/main/image/The_Squid_and_the_Whale_Thumbnail.jpg?raw=true"}
}
CLIP_NAMES = list(CLIP_VIDEO_MAPPING.keys())
CLOUDINARY_URL = "https://res.cloudinary.com/dlggzzrag/video/upload/heatmapoverlay_"

# === INITIALIZE SESSION STATE ===
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'selected_clip' not in st.session_state:
    st.session_state['selected_clip'] = None
if 'video_frame_idx' not in st.session_state:
    st.session_state['video_frame_idx'] = 0
if 'is_playing' not in st.session_state:
    st.session_state['is_playing'] = False

# === HELPER FUNCTIONS ===
def add_summ_navigation():
    """Adds a floating button at the bottom right corner for summary page navigation."""
    # Create container for the button with custom CSS
    button_container = st.container()
    
    # Apply CSS for positioning at bottom right
    button_css = """
    <style>
    .floating-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 100;
    }
    </style>
    """
    st.markdown(button_css, unsafe_allow_html=True)
    
    # Create the floating button with HTML
    with button_container:
        st.markdown(
            """
            <div class="floating-button">
                <form method="post" action="">
                    <input type="hidden" name="summ_nav" value="true">
                    <button type="submit" 
                        style="background-color: #4CAF50; color: white; padding: 10px 20px; 
                        border: none; border-radius: 5px; cursor: pointer;">
                        View Summary
                    </button>
                </form>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Check if the button was clicked (will be handled by session state)
    if st.session_state.get('summ_nav_clicked', False):
        st.session_state['summ_nav_clicked'] = False
        st.session_state['page'] = 'summ'
        st.rerun()

# === PAGE 1: Home (Clip Selection) ===
# In your home_page function, when a movie is selected:
def home_page():
    """Displays the clip selection page."""
    st.title("Seeing Stories: A Visual Analysis of Eye-Tracking Data across movie genres")
    st.write("Our eyes don't just watch -- they reveal. In this project, we explore how people visually engage with movies across different genres using eye-tracking data. From action-packed thrillers to emotion-rich dramas, we tracked every gaze, and every flicker of attention.")
    
    # Add a randomized prefix or timestamp to ensure keys are unique on each rerun
    session_id = st.session_state.get('session_id', str(int(time.time())))
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = session_id
    
    cols = st.columns(3)
    for i, clip_name in enumerate(CLIP_NAMES):
        # Use unique keys with session_id to prevent duplication
        img_path = CLIP_VIDEO_MAPPING[clip_name]["ThumbnailURL"]

        with cols[i % 3]:
            # Your image display code...
            st.image(img_path, use_container_width=True)
            if st.button(clip_name, key=f"{session_id}_clip_{i}", use_container_width=True):
                st.session_state['page'] = 'detail'
                st.session_state['selected_clip'] = clip_name
                st.session_state['video_frame_idx'] = 0
                st.session_state['is_playing'] = False
                st.session_state['light_condition'] = None
                st.rerun()
                st.session_state['page'] = 'detail'
                st.session_state['selected_clip'] = clip_name
                st.session_state['video_frame_idx'] = 0
                st.session_state['is_playing'] = False
                st.rerun()


# === PAGE 2: Clip Detail ===
def detail_page():
    """Displays the detail page for the selected clip with different visualization options."""
    selected_clip = st.session_state.get('selected_clip', '')
    
    # Initialize light condition if not already set
    if 'light_condition' not in st.session_state:
        st.session_state['light_condition'] = None
    
    # Header with clip title and back button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title(f"{selected_clip}")
    # with col2:
    #     if st.button("Back to Home", key="back_main"):
    #         st.session_state['page'] = 'home'
    #         st.rerun()
    
    # Light condition selection using segmented control
    light_condition = st.segmented_control(
        label = 'Light Conditions',
        options=["Low", "Medium", "High"],
        key="light_condition_segmented"
    )

    # Process the selection
    if light_condition and light_condition.lower() != st.session_state.get('light_condition', ''):
        st.session_state['light_condition'] = light_condition.lower()
        st.rerun()
    
    # Display current light condition
    if st.session_state['light_condition']:
        st.info(f"Currently viewing: {selected_clip} under {st.session_state['light_condition']} light conditions")
    else:
        st.warning("Please select a light condition to view the analysis")
        return  # Don't display tabs until a light condition is selected
    
    # Tabs for different visualizations based on light condition
    tab1, tab2, tab3 = st.tabs(["Pupil Size", "Heatmap", "Gaze Sequence"])
    
    light_condition = st.session_state['light_condition']
    
    with tab1:
        st.header(f"Pupil Size Analysis - {light_condition.upper()} Light")
        st.write("⏰Wait for the data to fill here🙌🏻🥶🫱🏻‍🫲🏾")
        
        # # Generate different data based on light condition
        # if light_condition == 'low':
        #     pupil_data = [random.randint(15, 25) for _ in range(100)]  # Smaller pupils in low light
        # elif light_condition == 'medium':
        #     pupil_data = [random.randint(20, 30) for _ in range(100)]  # Medium-sized pupils
        # else:  # high light
        #     pupil_data = [random.randint(25, 40) for _ in range(100)]  # Larger pupils in high light
            
        # st.line_chart({"Pupil Diameter": pupil_data})
        
    
    with tab2:
        st.header(f"Gaze Heatmap - {light_condition.upper()} Light")
        
        # Different heatmap visualization based on light condition
        if light_condition == 'low':
            # Import heatmap for low light
            clip_name = CLIP_VIDEO_MAPPING[selected_clip]['Low'][0]
            heatmap_url = (CLIP_VIDEO_MAPPING[selected_clip]['Low'][1]).strip("HEATMAPURL:")

            try:
                st.video(heatmap_url)    
            except FileNotFoundError:
                st.error(f"Video file not found: {heatmap_url}")
            except Exception as e:
                st.error(f"Error displaying video: {e}")


        elif light_condition == 'medium':
            # Import heatmap for medium light
            clip_name = CLIP_VIDEO_MAPPING[selected_clip]['Medium'][0]
            heatmap_url = (CLIP_VIDEO_MAPPING[selected_clip]['Medium'][1]).strip("HEATMAPURL:")

            try:
                st.video(heatmap_url)    
            except FileNotFoundError:
                st.error(f"Video file not found: {heatmap_url}")
            except Exception as e:
                st.error(f"Error displaying video: {e}")

        else:  # high light
            # Import heatmap for high light
            clip_name = CLIP_VIDEO_MAPPING[selected_clip]['High'][0]
            heatmap_url = (CLIP_VIDEO_MAPPING[selected_clip]['High'][1]).strip("HEATMAPURL:")

            try:
                st.video(heatmap_url)    
            except FileNotFoundError:
                st.error(f"Video file not found: {heatmap_url}")
            except Exception as e:
                st.error(f"Error displaying video: {e}")
        
    
    with tab3:
        st.header(f"Gaze Sequence - {light_condition.upper()} Light")
        st.write("⏰Wait for the data to fill here🙌🏻🥶🫱🏻‍🫲🏾")
        
        # # Different gaze data based on light condition
        # if light_condition == 'low':
        #     # More concentrated gaze points for low light
        #     x_pos = [random.randint(200, 300) for _ in range(20)]
        #     y_pos = [random.randint(100, 200) for _ in range(20)]
        # elif light_condition == 'medium':
        #     # Moderate spread for medium light
        #     x_pos = [random.randint(150, 350) for _ in range(20)]
        #     y_pos = [random.randint(75, 225) for _ in range(20)]
        # else:  # high light
        #     # Wider spread for high light
        #     x_pos = [random.randint(100, 400) for _ in range(20)]
        #     y_pos = [random.randint(50, 250) for _ in range(20)]
            
        # gaze_data = pd.DataFrame({
        #     'Time': list(range(20)),
        #     'X_Position': x_pos,
        #     'Y_Position': y_pos
        # })
        
        # st.dataframe(gaze_data)
        
        # st.scatter_chart(
        #     gaze_data,
        #     x='X_Position',
        #     y='Y_Position'
        # )

    # Add some space before the footer
    # st.markdown("<br><br>", unsafe_allow_html=True)
        
    # Create a footer container
    footer_container = st.container()
        
    with footer_container:
        # Create three columns: empty space on left, buttons in middle, empty space on right
        left_space, button_col1, button_col2, right_space = st.columns([2, 2, 2, 2])
            
        # Place the Back to Home button in the first button column
        with left_space:
            if st.button("Back to Home", key="back_btn", use_container_width=True):
                st.session_state['page'] = 'home'
                st.rerun()
            
        # Place the View Summary button in the second button column
        with right_space:
            if st.button("View Summary", key="summary_btn", use_container_width=True):
                st.session_state['page'] = 'summ'
                st.rerun()

# === PAGE 3: Clip Detail ===
def summ_page():
    """Displays a summary page with embedded Tableau dashboards."""
    
    st.title("⏰Wait for Gaze Analysis Summary Dashboard...")
    
    # Description
    st.write("This page shows comprehensive analysis dashboards created in Tableau.")
    st.write("But we are still waiting for the data to fill here🙌🏻😘🫱🏻‍🫲🏾")
    
    # # Create tabs for different dashboards
    # tab1, tab2 = st.tabs(["Overall Summary", "Detailed Metrics"])
    
    # with tab1:
    #     st.header("Overall Gaze Analysis")
    #     st.write("This dashboard shows aggregated data across all videos and light conditions.")
        
    #     # Replace the URL with your actual Tableau Public dashboard URL
    #     tableau_url = "https://public.tableau.com/views/YourOverallDashboard/Dashboard1?:embed=true&:showVizHome=no&:toolbar=no"
        
    #     # Embed the Tableau dashboard
    #     st.components.v1.iframe(
    #         src=tableau_url,
    #         width=1000,
    #         height=700,
    #         scrolling=False
    #     )
    
    # with tab2:
    #     st.header("Detailed Metrics by Movie and Light Condition")
        
    #     # Filter controls
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         # These would be your actual clip names
    #         selected_clip = st.selectbox(
    #             "Select Movie", 
    #             options=CLIP_NAMES,
    #             key="tableau_clip_select"
    #         )
        
    #     with col2:
    #         light_condition = st.selectbox(
    #             "Select Light Condition",
    #             options=["Low", "Medium", "High"],
    #             key="tableau_light_select" 
    #         )
        
    #     st.write(f"Showing detailed metrics for {selected_clip} under {light_condition} light conditions")
        
    #     # For a real implementation, you would dynamically construct a URL with filters
    #     # based on the selections above, something like:
    #     tableau_detail_url = f"https://public.tableau.com/views/YourDetailedDashboard/Dashboard2?:embed=true&:showVizHome=no&:toolbar=no&Movie={selected_clip}&Light={light_condition}"
        
    #     # Embed the filtered Tableau dashboard
    #     st.components.v1.iframe(
    #         src=tableau_detail_url,
    #         width=1000,
    #         height=700,
    #         scrolling=False
    #     )
        
    #     # Alternative embedding with HTML/JS API for more control
    #     # tableau_html = f"""
    #     # <div class='tableauPlaceholder' style='width: 1000px; height: 700px;'>
    #     #     <object class='tableauViz' width='1000' height='700' style='display:none;'>
    #     #         <param name='host_url' value='https://public.tableau.com/' />
    #     #         <param name='embed_code_version' value='3' />
    #     #         <param name='site_root' value='' />
    #     #         <param name='name' value='YourDetailedDashboard/Dashboard2' />
    #     #         <param name='tabs' value='no' />
    #     #         <param name='toolbar' value='yes' />
    #     #         <param name='filter' value='Movie={selected_clip}&Light={light_condition}' />
    #     #     </object>
    #     # </div>
    #     # """
    #     # st.components.v1.html(tableau_html, width=1020, height=720)

    # Create a footer container
    footer_container = st.container()
        
    with footer_container:
        # Create three columns: empty space on left, buttons in middle, empty space on right
        left_space, button_col1, button_col2, right_space = st.columns([2, 2, 2, 2])
            
    # Place the Back to Home button in the first button column
    with left_space:
        if st.button("Back to Home", key="back_btn", use_container_width=True):
            st.session_state['page'] = 'home'
            st.rerun()
            
    # Place the View Summary button in the second button column
    # with right_space:
    #     if st.button("View Summary", key="summary_btn", use_container_width=True):
    #         st.session_state['page'] = 'summ'
    #         st.rerun()


# === MAIN APP LOGIC ===
def main():
    # Initialize session state if needed
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'
    
    # Route to the correct page
    if st.session_state['page'] == 'home':
        home_page()
    elif st.session_state['page'] == 'detail':
        detail_page()
    elif st.session_state['page'] == 'summ':
        summ_page()

if __name__ == "__main__":
    main()
