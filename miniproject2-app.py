import streamlit as st
import numpy as np
import pandas as pd
import time
import cv2
import requests
import tempfile
import os

# === CONFIGURATION ===
NV_ROOT = '/Users/Tassanai/Library/CloudStorage/OneDrive-ChulalongkornUniversity/University/3Junior/Second Semester/2603278 INFO VISUAL/miniproject 2/dataset/NV'
CLIP_VIDEO_MAPPING =[{'Shrek': 
                        {
                            'Thumbnail': 'https://github.com/Tassanai5/material-miniproject2/blob/main/image/Shrek_Thumbnail.jpg?raw=true',
                            'Low': {
                                    'name': 'SHREK_10a',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443658/heatmapoverlay_SHREK_10a_s0qq7j.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
                                                                        ]
                                                    }
                                    },
                            'Medium': {
                                    'name': 'SHREK_3b',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443668/heatmapoverlay_SHREK_3b_lqcyyc.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
                                                                        ]
                                                    }
                                    },
                            'High': {
                                    'name': 'SHREK_3a',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443692/heatmapoverlay_SHREK_3a_dgrkux.mp4',
                                    'Gaze Sequence': {
                                                        'overall': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471343/shrek_overall_hukdzt.mp4',
                                                        'participant': ['https://res.cloudinary.com/dlggzzrag/video/upload/v1744471361/Shrekpath_2802ln_deeap9.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471360/Shrekpath_2792sl_k2tcpc.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471359/Shrekpath_2719sn_dqzmmw.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471357/Shrekpath_2103en_gvjqqe.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471357/Shrekpath_2086ty_bxmsh5.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471355/Shrekpath_2141po_po8v5g.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471350/Shrekpath_2039nn_s6jalp.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471347/Shrekpath_2024er_kxryht.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471346/Shrekpath_0811ne_aqvgby.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471345/Shrekpath_2017ae_uy4nhk.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471345/Shrekpath_2033kt_fw8pw7.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471342/Shrekpath_0724er_necb29.mp4']
                                                    }
                                                }
                        }
                    },
                    {'Deep Blue': 
                        {
                            'Thumbnail': 'https://github.com/Tassanai5/material-miniproject2/blob/main/image/Deep_Blue_Thumbnail.jpg?raw=true',
                            'Low': {
                                    'name': 'DEEPB_9c',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443688/heatmapoverlay_DEEPB_9c_uwtbxs.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
                                                                        ]
                                                    }
                                    },
                            'Medium': {
                                    'name': 'DEEPB_5b',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443488/heatmapoverlay_DEEPB_5b_p1erof.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
                                                                        ]
                                                    }
                                    },
                            'High': {
                                    'name': 'DEEPB_11a',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443720/heatmapoverlay_DEEPB_11a_packhk.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': ['',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '']
                                                    }
                                    }
                        }
                    },
                    {'The Squid and the Whale': 
                        {
                            'Thumbnail': 'https://github.com/Tassanai5/material-miniproject2/blob/main/image/The_Squid_and_the_Whale_Thumbnail.jpg?raw=true',
                            'Low': {
                                    'name': 'SQUID_8a',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443689/heatmapoverlay_SQUID_8a_vehssz.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
                                                                        ]
                                                    }
                                    },
                            'Medium': {
                                    'name': 'SQUID_12c',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443695/heatmapoverlay_SQUID_12c_x7zaou.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
                                                                        ]
                                                    }
                                    },
                            'High': {
                                    'name': 'SQUID_6a',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443691/heatmapoverlay_SQUID_6a_wpmqec.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': ['',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '']
                                                    }
                                    }
                        }
                    }]
CLIP_NAMES = ['Shrek', 'Deep Blue', 'The Squid and the Whale']

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

def display_frame_by_frame(video_url):
    """Display a video frame by frame with a slider control."""
    try:
        # Download the video from the URL
        if video_url.startswith(('http://', 'https://')):
            response = requests.get(video_url)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(response.content)
            temp_file.close()
            video_path = temp_file.name
        else:
            # If it's a local path
            video_path = video_url
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video: {video_url}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0:
            st.error("Could not determine frame count")
            return
        
        # Create a slider for frame selection
        frame_idx = st.slider("Frame", 0, total_frames - 1, 0)
        
        # Get timestamp for the current frame
        timestamp = frame_idx / fps
        st.text(f"Time: {timestamp:.2f}s (Frame {frame_idx+1}/{total_frames})")
        
        # Read the selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB (OpenCV uses BGR, but Streamlit expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            st.image(frame_rgb, caption=f"Frame {frame_idx+1}", use_column_width=True)
        else:
            st.error(f"Could not read frame {frame_idx}")
        
        # Add controls for playing a short sequence
        col1, col2 = st.columns(2)
        with col1:
            play_frames = st.button("Play 30 Frames")
        with col2:
            play_speed = st.select_slider("Speed", options=["0.5x", "1x", "2x"], value="1x")
        
        # Play a sequence of frames when the button is clicked
        if play_frames:
            speed_factor = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[play_speed]
            frame_container = st.empty()
            
            for i in range(frame_idx, min(frame_idx + 30, total_frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_container.image(frame_rgb, caption=f"Frame {i+1}", use_column_width=True)
                    time.sleep(1 / (fps * speed_factor))
                else:
                    break
        
        # Release the video capture
        cap.release()
        
    except Exception as e:
        st.error(f"Error displaying frame-by-frame view: {e}")
        # Fallback to standard video player
        st.write("Falling back to standard video player:")
        st.video(video_url)

def list_videos():
    videos = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
    overall_video = [v for v in videos if 'overall' in v.lower()]
    participant_videos = [v for v in videos if 'overall' not in v.lower()]
    return overall_video[0] if overall_video else None, participant_videos

def load_video(cloudinary_url):
    # Download the video data
    response = requests.get(cloudinary_url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Create a temporary file to store the video
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_video.write(chunk)
        temp_video_path = temp_video.name

    # Open the temporary file with cv2.VideoCapture
    cap = cv2.VideoCapture(temp_video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Delete the temporary file
    os.unlink(temp_video_path)

    return frames

def display_frame_by_frame(video_url):
    """Display a video frame by frame with a slider control."""
    
    # Create a unique key for this video in session state
    video_key = f"frames_{hash(video_url)}"
    
    # Check if we already have frames loaded for this video
    if video_key not in st.session_state:
        try:
            # Download the video from the URL
            if video_url.startswith(('http://', 'https://')):
                response = requests.get(video_url)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(response.content)
                temp_file.close()
                video_path = temp_file.name
            else:
                # If it's a local path
                video_path = video_url
            
            # Open the video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"Error opening video: {video_url}")
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0:
                st.error("Could not determine frame count")
                return
            
            # Pre-load frames (using a reasonable number to avoid memory issues)
            max_frames_to_load = min(total_frames, 300)  # Limit to 300 frames
            
            # Show loading message
            loading_msg = st.empty()
            loading_msg.info(f"Loading {max_frames_to_load} frames, please wait...")
            
            # Pre-load frames
            frames = []
            for i in range(max_frames_to_load):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    break
            
            # Store frames and video info in session state
            st.session_state[video_key] = {
                "frames": frames,
                "total_frames": total_frames,
                "fps": fps
            }
            
            # Clear loading message
            loading_msg.empty()
            
            # Release the video capture
            cap.release()
            
        except Exception as e:
            st.error(f"Error loading video frames: {e}")
            # Fallback to standard video player
            st.write("Falling back to standard video player:")
            st.video(video_url)
            return
    
    # Get the stored frames and video info from session state
    video_data = st.session_state[video_key]
    frames = video_data["frames"]
    total_frames = video_data["total_frames"]
    fps = video_data["fps"]
    
    # Create a slider for frame selection
    frame_idx = st.select_slider("Frame", 
                            options=[i for i in range(len(frames)) if i % 5 == 0] + [len(frames)-1], 
                            value=0)
    
    # Get timestamp for the current frame
    timestamp = frame_idx / fps
    st.text(f"Time: {timestamp:.2f}s (Frame {frame_idx}/{total_frames})")
    
    # Display frame container
    frame_container = st.empty()
    
    # Display the selected frame (from pre-loaded frames)
    if 0 <= frame_idx < len(frames):
        frame_container.image(frames[frame_idx], caption=f"Frame {frame_idx}", use_container_width=True)
    else:
        st.error(f"Frame index out of range: {frame_idx}")
    
    # Add controls for playing a short sequence
    col1, col2 , col3= st.columns([1, 1, 2])
    with col1:
        play_frames_45 = st.button("Play 45 Frames")
    with col2:
        play_frames_60 = st.button("Play 60 Frames")
    with col3:
        play_speed = st.select_slider("Speed", options=["0.5x", "1x", "2x"], value="1x")
    
    # Play a sequence of frames when the button is clicked
    if play_frames_45:
        speed_factor = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[play_speed]

        # Use the pre-loaded frames for playback
        end_idx = min(frame_idx + 45, len(frames))
        for i in range(frame_idx, end_idx, 1):  # Show every 5th frame
            if 0 <= i < len(frames):
                frame_container.image(frames[i], caption=f"Frame {i+1}", use_container_width=True)
                time.sleep(1 / (fps * speed_factor))

    elif play_frames_60:
        speed_factor = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[play_speed]

        # Use the pre-loaded frames for playback
        end_idx = min(frame_idx + 60, len(frames))
        for i in range(frame_idx, end_idx, 1):  # Show every 5th frame
            if 0 <= i < len(frames):
                frame_container.image(frames[i], caption=f"Frame {i+1}", use_container_width=True)
                time.sleep(1 / (fps * speed_factor))


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
        img_path = CLIP_VIDEO_MAPPING[i][clip_name]['Thumbnail']

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
    if selected_clip == 'Shrek':
        i = 0
    elif selected_clip == 'Deep Blue':
        i = 1
    elif selected_clip == 'The Squid and the Whale':
        i = 2
    
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
    if light_condition and light_condition != st.session_state.get('light_condition', ''):
        st.session_state['light_condition'] = light_condition
        st.rerun()

    # Display current light condition
    if st.session_state['light_condition']:
        st.info(f"Currently viewing: {selected_clip} under {st.session_state['light_condition']} light conditions")
    else:
        st.warning("Please select a light condition to view the analysis")
        
        # Add Back to Home button right after the warning
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:  # Center the button
            if st.button("Back to Home", key="back_btn", use_container_width=True):
                st.session_state['page'] = 'home'
                st.rerun()
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
        if light_condition == 'Low':
            # Import heatmap for low light
            clip_name = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['name']
            heatmap_url = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Heatmap']

            try:
                st.video(heatmap_url)    
            except FileNotFoundError:
                st.error(f"Video file not found: {heatmap_url}")
            except Exception as e:
                st.error(f"Error displaying video: {e}")


        elif light_condition == 'Medium':
            # Import heatmap for medium light
            clip_name = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['name']
            heatmap_url = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Heatmap']

            try:
                st.video(heatmap_url)    
            except FileNotFoundError:
                st.error(f"Video file not found: {heatmap_url}")
            except Exception as e:
                st.error(f"Error displaying video: {e}")

        else:  # high light
            # Import heatmap for high light
            clip_name = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['name']
            heatmap_url = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Heatmap']

            try:
                st.video(heatmap_url)    
            except FileNotFoundError:
                st.error(f"Video file not found: {heatmap_url}")
            except Exception as e:
                st.error(f"Error displaying video: {e}")
        
    
    with tab3:
        st.header(f"Gaze Sequence - {light_condition} Light")
        
        if 'path_type' not in st.session_state:
            st.session_state['path_type'] = None

        # Path type selection using segmented control
        path_type = st.segmented_control(
            label = 'Path Type',
            options=["Overall", "Individual"],
            key="path_type_segmented"
        )

        # Process the selection
        if path_type and path_type != st.session_state.get('path_type', ''):
            st.session_state['path_type'] = path_type
            st.rerun()

        # Display current light condition
        if not st.session_state['path_type']:
            st.warning("Please select a type to view the analysis")

        if path_type == 'Overall':
            gaze_path = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Gaze Sequence']['overall']

            display_frame_by_frame(gaze_path)


        elif path_type == 'Individual':
            gaze_path_list = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Gaze Sequence']['participant']
            n = len(gaze_path_list)
            abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
            participant_lst = [f"Participant {abc[i]}" for i in range(n)]
            participant = st.selectbox(
                                        "Who do you want to watch?",
                                        (participant_lst),
                                        index=None,
                                        placeholder="Select Participant..."
                                    )
            if participant:
                st.write("You selected:", participant)
                parti_no = abc.index(participant[-1])
                gaze_path = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Gaze Sequence']['participant'][parti_no]

                display_frame_by_frame(gaze_path)
            
            else:
                st.warning("Please select participant")

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
