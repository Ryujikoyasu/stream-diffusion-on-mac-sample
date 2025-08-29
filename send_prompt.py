import socket
import threading
import sys

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

def receive_messages(sock):
    """Function to continuously receive and display messages from the server"""
    while True:
        try:
            data = sock.recv(1024)
            if not data:
                break
            print(f"\n[Server] {data.decode('utf-8')}")
            print("\nEnter prompt (or press Ctrl+C to exit): ", end="")
            sys.stdout.flush()  # Ensure the prompt is displayed
        except:
            # If any error occurs, break the loop
            break

print("\n===== StreamDiffusion Prompt Client =====")
print("Connect to the StreamDiffusion server and send prompts.")
print("The server will enhance your prompts with LLM and generate images.")
print("Type a prompt and press Enter to send it.")
print("Press Ctrl+C to exit.")
print("=======================================\n")

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Set a timeout for the initial connection attempt
        s.settimeout(5)
        try:
            s.connect((HOST, PORT))
            print(f"Connected to StreamDiffusion server at {HOST}:{PORT}")
            # Reset timeout for normal operation
            s.settimeout(None)
            
            # Start a thread to receive messages from the server
            receiver_thread = threading.Thread(target=receive_messages, args=(s,))
            receiver_thread.daemon = True  # Thread will exit when main program exits
            receiver_thread.start()
            
            while True:
                prompt = input("Enter prompt (or press Ctrl+C to exit): ")
                s.sendall(prompt.encode("utf-8"))
        except socket.timeout:
            print("Connection timed out. Is the StreamDiffusion server running?")
        except ConnectionRefusedError:
            print("Connection refused. Is the StreamDiffusion server running?")
        except Exception as e:
            print(f"Error: {e}")
            
except KeyboardInterrupt:
    print("\nExiting prompt client. Goodbye!")