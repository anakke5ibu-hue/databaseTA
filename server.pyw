import customtkinter as ctk
import subprocess
import os

# Konfigurasi Tema CustomTkinter
ctk.set_appearance_mode("Dark")  # Bisa diganti "Light" atau "System"
ctk.set_default_color_theme("blue")

class ServerManagerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup Window
        self.title("Panel Kontrol Server Parkir")
        self.geometry("350x250")
        self.resizable(False, False)

        # Variabel untuk menyimpan proses server
        self.server_process = None

        # --- UI Elements ---
        
        # Judul
        self.title_label = ctk.CTkLabel(self, text="Server Dashboard", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=(20, 10))

        # Status Label
        self.status_label = ctk.CTkLabel(self, text="Status: OFFLINE", text_color="red", font=ctk.CTkFont(size=14, weight="bold"))
        self.status_label.pack(pady=(0, 20))

        # Tombol Nyalakan Server
        self.btn_start = ctk.CTkButton(self, text="▶ Nyalain Server", fg_color="green", hover_color="darkgreen", command=self.start_server)
        self.btn_start.pack(pady=10)

        # Tombol Matikan Server
        self.btn_stop = ctk.CTkButton(self, text="⏹ Matiin Server", fg_color="red", hover_color="darkred", command=self.stop_server, state="disabled")
        self.btn_stop.pack(pady=10)

    def start_server(self):
        if self.server_process is None:
            # Ambil path direktori
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Path ke python.exe di venv_parkir
            python_exe = os.path.join(base_dir, "venv_parkir", "Scripts", "python.exe")
            server_script = os.path.join(base_dir, "main.py") # Ganti nama file jika beda

            try:
                # Menjalankan server menggunakan subprocess
                # Gunakan CREATE_NO_WINDOW agar console server tidak muncul
                self.server_process = subprocess.Popen(
                    [python_exe, server_script],
                    cwd=base_dir,
                    creationflags=subprocess.CREATE_NO_WINDOW 
                )

                # Update UI
                self.status_label.configure(text="Status: ONLINE", text_color="lightgreen")
                self.btn_start.configure(state="disabled", fg_color="gray")
                self.btn_stop.configure(state="normal", fg_color="red")
                
            except Exception as e:
                self.status_label.configure(text=f"Error: {str(e)}", text_color="red")

    def stop_server(self):
        if self.server_process is not None:
            # Matikan proses server
            self.server_process.terminate()
            self.server_process = None

            # Update UI
            self.status_label.configure(text="Status: OFFLINE", text_color="red")
            self.btn_start.configure(state="normal", fg_color="green")
            self.btn_stop.configure(state="disabled", fg_color="gray")

    # Fungsi yang dipanggil saat aplikasi (X) ditutup
    def on_closing(self):
        # Pastikan server dimatikan saat window di-close
        self.stop_server()
        self.destroy()

if __name__ == "__main__":
    app = ServerManagerApp()
    # Daftarkan protokol saat tombol close / X di klik
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()