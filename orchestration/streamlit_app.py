import streamlit as st
import sys
from pathlib import Path
import time
import redis
import json
import subprocess
import tempfile # Added for temporary file handling
from faker import Faker # Import Faker

# Initialize Faker
fake = Faker()

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="LangGraph Workflow Monitor", layout="wide")

# --- Script Configuration ---
# Get the directory of the Streamlit app.
# We assume langGraphRun.py and the core_Agents directory are in the same directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PATH_TO_LANGGRAPH_SCRIPT = str(SCRIPT_DIR / "langGraphRun.py")
# Directory for temporary uploads by Streamlit, will be created if it doesn't exist
TEMP_UPLOAD_DIR_STREAMLIT = SCRIPT_DIR / "temp_streamlit_uploads"


# --- Redis Configuration for Streamlit App ---
redis_client_streamlit = None
redis_connection_error = None # Initialize to None
try:
    redis_client_streamlit = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client_streamlit.ping()
except redis.exceptions.ConnectionError as e:
    print(f"Streamlit App: Could not connect to Redis: {e}")
    redis_connection_error = f"Streamlit App: Could not connect to Redis: {e}. Log display from Redis will not be available."
except Exception as e:
    print(f"Streamlit App: An unexpected error occurred during Redis connection: {e}")
    redis_connection_error = f"Streamlit App: An unexpected error occurred during Redis connection: {e}. Log display from Redis will not be available."


def get_orchestration_log_for_streamlit(orchestration_id: str) -> list:
    """
    Fetches and parses the orchestration log from Redis for a given orchestration_id.
    """
    if not redis_client_streamlit:
        print("Redis client not connected in Streamlit app (get_orchestration_log).")
        return [{"error": "Redis client not connected in Streamlit app."}]
    try:
        log_list_key = f"orchestration_log:{orchestration_id}"
        entry_keys = redis_client_streamlit.zrange(log_list_key, 0, -1)
        log_entries = []
        for key in entry_keys:
            entry_data = redis_client_streamlit.hgetall(key)
            if entry_data:
                for k, v in entry_data.items():
                    if isinstance(v, str):
                        if (v.startswith('{') and v.endswith('}')) or \
                           (v.startswith('[') and v.endswith(']')):
                            try:
                                entry_data[k] = json.loads(v)
                            except json.JSONDecodeError:
                                pass 
                log_entries.append(entry_data)
        return log_entries
    except redis.exceptions.RedisError as e:
        print(f"Streamlit App: Could not read orchestration log from Redis for {orchestration_id}: {e}")
        return [{"error": f"Redis read error for {orchestration_id}: {e}"}]
    except Exception as e:
        print(f"Streamlit App: An unexpected error occurred while reading log for {orchestration_id}: {e}")
        return [{"error": f"Unexpected log read error for {orchestration_id}: {e}"}]

# Function to generate fake JSON data
def generate_fake_json_data():
    return {
        "event": fake.random_element(elements=('user_registered', 'product_viewed', 'order_placed', 'item_added_to_cart')),
        "timestamp": fake.date_time_this_year().isoformat(),
        "user_id": fake.uuid4(),
        "product_id": fake.isbn13(),
        "quantity": fake.random_int(min=1, max=5),
        "price": float(round(fake.random_number(digits=3, fix_len=False) + fake.pydecimal(left_digits=2, right_digits=2, positive=True), 2)), # Convert Decimal to float
        "currency": fake.currency_code(),
        "user_agent": fake.user_agent(),
        "ip_address": fake.ipv4(),
        "referrer": fake.uri(),
        "details": {
            "name": fake.name(),
            "email": fake.email(),
            "address": fake.address()
        }
    }

# --- Streamlit App UI ---
st.title("LangGraph Workflow Monitor")

if redis_connection_error:
    st.sidebar.error(redis_connection_error)
elif redis_client_streamlit:
    st.sidebar.success("Successfully connected to Redis!")

st.sidebar.header("Workflow Input")

# Add a button to generate fake JSON data
if st.sidebar.button("✨ Generate Fake JSON Data", key="generate_fake_json_button", use_container_width=True):
    fake_json = generate_fake_json_data()
    st.session_state.raw_input_data_text = json.dumps(fake_json, indent=2)


raw_input_data_text = st.sidebar.text_area(
    "Paste raw input data (Email, JSON text):",
    value=st.session_state.get("raw_input_data_text", ""), # Use session state for persistence
    height=250, # Adjusted height
    key="raw_input_data_area",
    help="Textual input like email content or a JSON string."
)


st.sidebar.header("Or Upload File")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or Text file", 
    type=["pdf", "txt", "json"], # Allow various file types
    key="file_uploader"
)

# Initialize session state variables
if 'process_output' not in st.session_state:
    st.session_state.process_output = ""
if 'orchestration_id' not in st.session_state:
    st.session_state.orchestration_id = None
if 'run_triggered' not in st.session_state:
    st.session_state.run_triggered = False
if 'log_data' not in st.session_state:
    st.session_state.log_data = []

if st.sidebar.button("🚀 Run Workflow", key="run_workflow_button_main", use_container_width=True):
    input_for_script = None
    input_type_for_script = "auto"  # langGraphRun.py will auto-detect if not 'pdf'
    temp_file_path_to_delete = None

    # Create the temporary directory for uploads by Streamlit if it doesn't exist
    TEMP_UPLOAD_DIR_STREAMLIT.mkdir(exist_ok=True)

    if uploaded_file is not None:
        # Save uploaded file to a temporary file within our defined temp dir
        # This makes cleanup easier and keeps temp files organized.
        suffix = Path(uploaded_file.name).suffix or ".dat" # Get original suffix or use .dat
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_UPLOAD_DIR_STREAMLIT) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            input_for_script = tmp_file.name  # Pass the path to the script
            temp_file_path_to_delete = tmp_file.name # Keep track for deletion

        # Determine input_type_for_script based on uploaded file extension
        if uploaded_file.type == "application/pdf" or (isinstance(input_for_script, str) and input_for_script.lower().endswith(".pdf")):
            input_type_for_script = "pdf"
        elif uploaded_file.type == "application/json" or (isinstance(input_for_script, str) and input_for_script.lower().endswith(".json")):
            # If it's a JSON file, langGraphRun will read its content.
            # The preprocess_input_for_agents in langGraphRun will handle it as a string.
            input_type_for_script = "json_file" # Custom hint for streamlit to pass path
        elif uploaded_file.type == "text/plain" or (isinstance(input_for_script, str) and input_for_script.lower().endswith(".txt")):
            # For .txt files, it could be email or other text.
            # langGraphRun.py's preprocess will treat its content as a string.
            input_type_for_script = "text_file"
        else:
            input_type_for_script = "auto" # Fallback for other file types

        st.sidebar.info(f"Using uploaded file: {uploaded_file.name} (Type hint: {input_type_for_script})")
        # Optionally clear text area if file is used, or allow both (current behavior lets script decide)
        # if raw_input_data_text:
        # st.sidebar.warning("File uploaded. Text area input will be ignored by the script if --input is a file path.")
    
    elif raw_input_data_text:
        input_for_script = raw_input_data_text
        # Heuristic for text area content for --type hint, 'auto' is default in script anyway
        if (raw_input_data_text.strip().startswith("{") and raw_input_data_text.strip().endswith("}")) or \
           (raw_input_data_text.strip().startswith("[") and raw_input_data_text.strip().endswith("]")):
            input_type_for_script = "json" # Hint that it's a JSON string
        else:
            input_type_for_script = "email" # Default hint for other raw text
        st.sidebar.info(f"Using text area input (Type hint: {input_type_for_script}).")
    else:
        st.sidebar.warning("⚠️ Please provide input data in the text area or upload a file.")
        input_for_script = None # Ensure it's None to prevent running

    if input_for_script:
        st.session_state.process_output = ""
        st.session_state.orchestration_id = None
        st.session_state.log_data = []
        st.session_state.run_triggered = True

        processing_placeholder = st.empty()
        with processing_placeholder.container():
            st.info(f"⚙️ Processing workflow with input type hint: {input_type_for_script}... Please wait.")
            with st.spinner("LangGraph workflow is running..."):
                try:
                    cmd = [
                        sys.executable, PATH_TO_LANGGRAPH_SCRIPT,
                        "--input", str(input_for_script), # Ensure path/text is string
                    ]
                    # langGraphRun.py's argparse default for --type is 'auto'.
                    # We only pass --type if we have a specific hint from Streamlit,
                    # especially 'pdf' for PDF paths.
                    # For json_file or text_file, langGraphRun.py will read the file path content.
                    if input_type_for_script in ["pdf", "json", "email"]: # Specific hints for script
                         cmd.extend(["--type", input_type_for_script])
                    # If input_type_for_script is 'json_file' or 'text_file',
                    # langGraphRun's 'auto' type detection will handle the file path.

                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=SCRIPT_DIR # Essential for relative paths in langGraphRun.py
                    )

                    stdout_full = ""
                    for line in iter(process.stdout.readline, ''):
                        stdout_full += line
                        if "STARTING ORCHESTRATION" in line and not st.session_state.orchestration_id:
                            try:
                                parts = line.split("STARTING ORCHESTRATION")
                                if len(parts) > 1:
                                    oid_part = parts[1].split("=====")[0].strip()
                                    st.session_state.orchestration_id = oid_part
                                    st.toast(f"Workflow started! ID: {st.session_state.orchestration_id}", icon="🎉")
                            except IndexError:
                                st.toast("Error parsing orchestration ID from live output.", icon="⚠️")
                    
                    remaining_stdout, stderr_output = process.communicate(timeout=120) # Added timeout
                    stdout_full += remaining_stdout
                    st.session_state.process_output = stdout_full

                    if process.returncode == 0:
                        st.success("✅ Workflow script finished successfully.")
                        if not st.session_state.orchestration_id and "STARTING ORCHESTRATION" in stdout_full:
                            try:
                                parts = stdout_full.split("STARTING ORCHESTRATION")
                                if len(parts) > 1:
                                     st.session_state.orchestration_id = parts[1].split("=====")[0].strip()
                                     st.toast(f"Retrieved Orchestration ID: {st.session_state.orchestration_id}", icon="🆔")
                            except Exception:
                                st.warning("Could not automatically parse Orchestration ID from script's full output.")
                        elif st.session_state.orchestration_id:
                             # Displaying the ID if found, for user reference
                             st.markdown(f"**Processed Orchestration ID:** `{st.session_state.orchestration_id}`")
                    else:
                        st.error(f"❌ Workflow script failed with return code {process.returncode}.")
                        st.session_state.process_output += "\n--- STDERR ---\n" + stderr_output
                    
                    processing_placeholder.empty()

                except subprocess.TimeoutExpired:
                    st.error("Workflow script timed out after 120 seconds.")
                    process.kill()
                    _, stderr_output = process.communicate()
                    st.session_state.process_output += "\n--- TIMEOUT ERROR ---\nScript execution exceeded timeout."
                    if stderr_output:
                        st.session_state.process_output += "\n--- STDERR (after timeout) ---\n" + stderr_output
                    processing_placeholder.empty()
                except FileNotFoundError:
                    st.error(f"Critical Error: The script '{PATH_TO_LANGGRAPH_SCRIPT}' was not found.")
                    processing_placeholder.empty()
                except Exception as e:
                    st.error(f"An error occurred while trying to run the workflow: {str(e)}")
                    st.session_state.process_output += f"\n--- Streamlit App Error ---\n{str(e)}"
                    processing_placeholder.empty()
                finally:
                    # Clean up the temporary file if one was created
                    if temp_file_path_to_delete and Path(temp_file_path_to_delete).exists():
                        try:
                            Path(temp_file_path_to_delete).unlink()
                            print(f"[STREAMLIT_CLEANUP] Deleted temp file: {temp_file_path_to_delete}")
                        except Exception as e_clean:
                            print(f"[STREAMLIT_CLEANUP_ERROR] Could not delete temp file {temp_file_path_to_delete}: {e_clean}")
        
        if st.session_state.orchestration_id or st.session_state.process_output:
            st.rerun()


# --- Main Display Area for Logs and Output ---
col1, col2 = st.columns(2)

with col1:
    st.header("📝 Workflow Log (from Redis)")
    if not redis_client_streamlit and not redis_connection_error:
        st.warning("Redis client is not available. Logs cannot be fetched.")
    
    if st.session_state.orchestration_id:
        if not redis_client_streamlit and redis_connection_error : # Show error again if trying to use OID without redis
             st.error(redis_connection_error)

        st.markdown(f"**Orchestration ID:** `{st.session_state.orchestration_id}`")
        
        if st.button("🔄 Refresh Log", key="refresh_log_button", disabled=not redis_client_streamlit):
            if redis_client_streamlit and st.session_state.orchestration_id:
                st.session_state.log_data = get_orchestration_log_for_streamlit(st.session_state.orchestration_id)
                if not st.session_state.log_data or (len(st.session_state.log_data) == 1 and "error" in st.session_state.log_data[0]):
                    st.toast("No new log entries found or Redis not updated yet.", icon="📭")
                elif any("error" in entry for entry in st.session_state.log_data):
                     st.error("An error occurred while fetching logs from Redis. See console or previous messages.")
                else:
                    st.toast(f"Fetched {len(st.session_state.log_data)} log entries.", icon="📬")
            elif not redis_client_streamlit:
                 st.warning("Cannot refresh log: Redis client not connected.")

        if st.session_state.run_triggered and not st.session_state.log_data and redis_client_streamlit and st.session_state.orchestration_id:
             st.session_state.log_data = get_orchestration_log_for_streamlit(st.session_state.orchestration_id)
             st.session_state.run_triggered = False 
             if any("error" in entry for entry in st.session_state.log_data):
                 st.error("An error occurred while fetching initial logs. Check console or previous messages.")

        if st.session_state.log_data:
            display_logs = [entry for entry in st.session_state.log_data if "error" not in entry]
            error_logs = [entry for entry in st.session_state.log_data if "error" in entry]

            if error_logs:
                for err_entry in error_logs:
                    st.error(f"Log fetching error: {err_entry.get('error')}")
            
            if display_logs:
                st.write(f"Displaying {len(display_logs)} log entries:")
                for i, entry in enumerate(display_logs):
                    entry_title_key = entry.get('item_key', entry.get('agent_name', entry.get('triggered_action', f"Log Entry {i + 1}")))
                    with st.expander(f"{entry.get('timestamp', '')} - {entry_title_key}", expanded=(i == len(display_logs) - 1)):
                        st.json(entry)
            elif not error_logs and st.session_state.orchestration_id:
                 st.info("No valid log entries found in Redis for this ID.")

        elif st.session_state.orchestration_id and redis_client_streamlit:
            st.info("No log entries found for this ID yet. Click 'Refresh Log' or check if the workflow produced logs.")
        elif not st.session_state.orchestration_id : # No OID yet
             st.info(" Run a workflow to see its detailed log here.")
             if redis_connection_error: # Remind about redis issue if no OID yet
                st.error(f"Note: {redis_connection_error}")


    else: # No orchestration_id in session state yet
        st.info(" Run a workflow to see its detailed log here.")
        if redis_connection_error:
             st.error(f"Note: {redis_connection_error}")


with col2:
    st.header("📜 Raw Script Output (stdout/stderr)")
    if st.session_state.process_output:
        st.text_area(
            "Output from langGraphRun.py:",
            value=st.session_state.process_output,
            height=500,
            disabled=True,
            key="script_output_display_area"
        )
    else:
        st.info(" No script output to display. Run a workflow.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed with LangGraph & Streamlit.")