import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
import threading
from collections import deque
import heapq
from queue import PriorityQueue

class ProbabilityDistributionAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Application setup
        self.title("Probability Distribution Analyzer")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        # Set theme and style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Configure colors
        self.bg_color = "#f5f5f5"
        self.accent_color = "#4a6fa5"
        self.text_color = "#333333"
        
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TButton", 
                            background=self.accent_color, 
                            foreground="white", 
                            padding=10, 
                            font=("Helvetica", 10))
        self.style.map("TButton",
                      background=[("active", "#3a5985")])
        self.style.configure("TLabel", 
                            background=self.bg_color, 
                            foreground=self.text_color,
                            font=("Helvetica", 11))
        self.style.configure("Header.TLabel", 
                            background=self.bg_color, 
                            foreground=self.accent_color,
                            font=("Helvetica", 16, "bold"))
        self.style.configure("TNotebook", background=self.bg_color)
        self.style.configure("TNotebook.Tab", 
                            background="#e0e0e0", 
                            foreground=self.text_color,
                            padding=[15, 5],
                            font=("Helvetica", 10))
        self.style.map("TNotebook.Tab",
                      background=[("selected", self.accent_color)],
                      foreground=[("selected", "white")])
                      
        self.configure(bg=self.bg_color)
        
        # Data structures for storing datasets
        self.datasets = {}  # Dictionary to store loaded datasets
        self.analysis_cache = {}  # Cache for analysis results
        self.recent_files = deque(maxlen=5)  # Store recent files
        
        # Create main container
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create top panel for file operations
        self.create_top_panel()
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs
        self.create_data_view_tab()
        self.create_normal_distribution_tab()
        self.create_poisson_distribution_tab()
        self.create_comparison_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Try to load default files
        self.default_paths = [
            r"D:\poisson calc\height_distribution.csv",
            r"D:\poisson calc\traffic_arrival_rates.csv"
        ]
        self.try_load_default_files()
    
    def create_top_panel(self):
        top_panel = ttk.Frame(self.main_container)
        top_panel.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(top_panel, text="Probability Distribution Analyzer", style="Header.TLabel")
        title_label.pack(side=tk.LEFT, padx=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(top_panel)
        buttons_frame.pack(side=tk.RIGHT)
        
        # Load data button
        load_button = ttk.Button(buttons_frame, text="Load Dataset", command=self.load_dataset)
        load_button.pack(side=tk.LEFT, padx=5)
        
        # Export results button
        export_button = ttk.Button(buttons_frame, text="Export Results", command=self.export_results)
        export_button.pack(side=tk.LEFT, padx=5)
        
        # Help button
        help_button = ttk.Button(buttons_frame, text="Help", command=self.show_help)
        help_button.pack(side=tk.LEFT, padx=5)
    
    def create_data_view_tab(self):
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data View")
        
        # Dataset selection frame
        selection_frame = ttk.Frame(self.data_frame)
        selection_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(selection_frame, text="Select Dataset:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(selection_frame, textvariable=self.dataset_var, state="readonly")
        self.dataset_combo.pack(side=tk.LEFT, padx=5)
        self.dataset_combo.bind("<<ComboboxSelected>>", self.update_data_view)
        
        # Statistics frame
        self.stats_frame = ttk.Frame(self.data_frame)
        self.stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Data view frame with scrollbar
        data_view_container = ttk.Frame(self.data_frame)
        data_view_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview with scrollbars
        self.tree_frame = ttk.Frame(data_view_container)
        self.tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree_scroll_y = ttk.Scrollbar(self.tree_frame)
        self.tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree_scroll_x = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL)
        self.tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree = ttk.Treeview(self.tree_frame, 
                             yscrollcommand=self.tree_scroll_y.set,
                             xscrollcommand=self.tree_scroll_x.set)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        self.tree_scroll_y.config(command=self.tree.yview)
        self.tree_scroll_x.config(command=self.tree.xview)
    
    def create_normal_distribution_tab(self):
        self.normal_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.normal_frame, text="Normal Distribution")
        
        # Controls frame
        controls_frame = ttk.Frame(self.normal_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(controls_frame, text="Dataset:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.normal_dataset_var = tk.StringVar()
        self.normal_dataset_combo = ttk.Combobox(controls_frame, textvariable=self.normal_dataset_var, state="readonly")
        self.normal_dataset_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controls_frame, text="Column:").pack(side=tk.LEFT, padx=(20, 10))
        
        self.normal_column_var = tk.StringVar()
        self.normal_column_combo = ttk.Combobox(controls_frame, textvariable=self.normal_column_var, state="readonly")
        self.normal_column_combo.pack(side=tk.LEFT, padx=5)
        
        analyze_button = ttk.Button(controls_frame, text="Analyze", command=self.analyze_normal)
        analyze_button.pack(side=tk.LEFT, padx=(20, 5))
        
        # Bind events
        self.normal_dataset_combo.bind("<<ComboboxSelected>>", self.update_normal_columns)
        
        # Results frame
        self.normal_results_frame = ttk.Frame(self.normal_frame)
        self.normal_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split into left and right panels
        left_panel = ttk.Frame(self.normal_results_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_panel = ttk.Frame(self.normal_results_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Stats frame on left
        stats_frame = ttk.LabelFrame(left_panel, text="Distribution Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.normal_stats_text = tk.Text(stats_frame, height=10, width=40, wrap=tk.WORD)
        self.normal_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.normal_stats_text.config(state=tk.DISABLED)
        
        # Goodness of fit frame
        gof_frame = ttk.LabelFrame(left_panel, text="Goodness of Fit")
        gof_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.normal_gof_text = tk.Text(gof_frame, height=8, width=40, wrap=tk.WORD)
        self.normal_gof_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.normal_gof_text.config(state=tk.DISABLED)
        
        # Figure frame on right
        self.normal_fig_frame = ttk.Frame(right_panel)
        self.normal_fig_frame.pack(fill=tk.BOTH, expand=True)
    
    def create_poisson_distribution_tab(self):
        self.poisson_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.poisson_frame, text="Poisson Distribution")
        
        # Controls frame
        controls_frame = ttk.Frame(self.poisson_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(controls_frame, text="Dataset:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.poisson_dataset_var = tk.StringVar()
        self.poisson_dataset_combo = ttk.Combobox(controls_frame, textvariable=self.poisson_dataset_var, state="readonly")
        self.poisson_dataset_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controls_frame, text="Column:").pack(side=tk.LEFT, padx=(20, 10))
        
        self.poisson_column_var = tk.StringVar()
        self.poisson_column_combo = ttk.Combobox(controls_frame, textvariable=self.poisson_column_var, state="readonly")
        self.poisson_column_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controls_frame, text="Time Interval:").pack(side=tk.LEFT, padx=(20, 10))
        
        self.time_interval_var = tk.StringVar(value="1")
        time_interval_entry = ttk.Entry(controls_frame, textvariable=self.time_interval_var, width=5)
        time_interval_entry.pack(side=tk.LEFT, padx=5)
        
        analyze_button = ttk.Button(controls_frame, text="Analyze", command=self.analyze_poisson)
        analyze_button.pack(side=tk.LEFT, padx=(20, 5))
        
        # Bind events
        self.poisson_dataset_combo.bind("<<ComboboxSelected>>", self.update_poisson_columns)
        
        # Results frame
        self.poisson_results_frame = ttk.Frame(self.poisson_frame)
        self.poisson_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split into left and right panels
        left_panel = ttk.Frame(self.poisson_results_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_panel = ttk.Frame(self.poisson_results_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Stats frame on left
        stats_frame = ttk.LabelFrame(left_panel, text="Distribution Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.poisson_stats_text = tk.Text(stats_frame, height=10, width=40, wrap=tk.WORD)
        self.poisson_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.poisson_stats_text.config(state=tk.DISABLED)
        
        # Goodness of fit frame
        gof_frame = ttk.LabelFrame(left_panel, text="Goodness of Fit")
        gof_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.poisson_gof_text = tk.Text(gof_frame, height=8, width=40, wrap=tk.WORD)
        self.poisson_gof_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.poisson_gof_text.config(state=tk.DISABLED)
        
        # Figure frame on right
        self.poisson_fig_frame = ttk.Frame(right_panel)
        self.poisson_fig_frame.pack(fill=tk.BOTH, expand=True)
    
    def create_comparison_tab(self):
        self.comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_frame, text="Comparison")
        
        # Controls frame
        controls_frame = ttk.Frame(self.comparison_frame)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # First dataset selection
        ttk.Label(controls_frame, text="Dataset 1:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.comp_dataset1_var = tk.StringVar()
        self.comp_dataset1_combo = ttk.Combobox(controls_frame, textvariable=self.comp_dataset1_var, state="readonly", width=15)
        self.comp_dataset1_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(controls_frame, text="Column:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.comp_column1_var = tk.StringVar()
        self.comp_column1_combo = ttk.Combobox(controls_frame, textvariable=self.comp_column1_var, state="readonly", width=10)
        self.comp_column1_combo.pack(side=tk.LEFT, padx=(0, 20))
        
        # Second dataset selection
        ttk.Label(controls_frame, text="Dataset 2:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.comp_dataset2_var = tk.StringVar()
        self.comp_dataset2_combo = ttk.Combobox(controls_frame, textvariable=self.comp_dataset2_var, state="readonly", width=15)
        self.comp_dataset2_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(controls_frame, text="Column:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.comp_column2_var = tk.StringVar()
        self.comp_column2_combo = ttk.Combobox(controls_frame, textvariable=self.comp_column2_var, state="readonly", width=10)
        self.comp_column2_combo.pack(side=tk.LEFT, padx=(0, 20))
        
        # Compare button
        compare_button = ttk.Button(controls_frame, text="Compare", command=self.compare_distributions)
        compare_button.pack(side=tk.LEFT, padx=5)
        
        # Bind events
        self.comp_dataset1_combo.bind("<<ComboboxSelected>>", lambda e: self.update_comparison_columns(1))
        self.comp_dataset2_combo.bind("<<ComboboxSelected>>", lambda e: self.update_comparison_columns(2))
        
        # Results frame
        self.comparison_results_frame = ttk.Frame(self.comparison_frame)
        self.comparison_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top panel for figure
        self.comp_fig_frame = ttk.Frame(self.comparison_results_frame)
        self.comp_fig_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom panel for statistics
        stats_frame = ttk.LabelFrame(self.comparison_results_frame, text="Comparison Statistics")
        stats_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        self.comp_stats_text = tk.Text(stats_frame, height=8, width=40, wrap=tk.WORD)
        self.comp_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.comp_stats_text.config(state=tk.DISABLED)
    
    def try_load_default_files(self):
        """Try to load default datasets if they exist"""
        for path in self.default_paths:
            if os.path.exists(path):
                try:
                    filename = os.path.basename(path)
                    df = pd.read_csv(path)
                    self.datasets[filename] = df
                    self.recent_files.append(path)
                    self.status_var.set(f"Loaded default file: {filename}")
                except Exception as e:
                    print(f"Error loading default file {path}: {e}")
        
        self.update_dataset_combos()
    
    def update_dataset_combos(self):
        """Update all dataset combo boxes with current datasets"""
        dataset_names = list(self.datasets.keys())
        
        self.dataset_combo['values'] = dataset_names
        self.normal_dataset_combo['values'] = dataset_names
        self.poisson_dataset_combo['values'] = dataset_names
        self.comp_dataset1_combo['values'] = dataset_names
        self.comp_dataset2_combo['values'] = dataset_names
        
        if dataset_names:
            self.dataset_var.set(dataset_names[0])
            self.normal_dataset_var.set(dataset_names[0])
            self.poisson_dataset_var.set(dataset_names[0])
            self.comp_dataset1_var.set(dataset_names[0])
            if len(dataset_names) > 1:
                self.comp_dataset2_var.set(dataset_names[1])
            else:
                self.comp_dataset2_var.set(dataset_names[0])
            
            # Update views
            self.update_data_view()
            self.update_normal_columns()
            self.update_poisson_columns()
            self.update_comparison_columns(1)
            self.update_comparison_columns(2)
    
    def update_data_view(self, event=None):
        """Update the data view tab with selected dataset"""
        selected = self.dataset_var.get()
        if not selected or selected not in self.datasets:
            return
        
        df = self.datasets[selected]
        
        # Update statistics
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        # Basic statistics
        shape_label = ttk.Label(self.stats_frame, text=f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        shape_label.pack(side=tk.LEFT, padx=10)
        
        # Show data in treeview
        self.tree.delete(*self.tree.get_children())
        
        # Configure columns
        self.tree['columns'] = list(df.columns)
        self.tree['show'] = 'headings'
        
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)
        
        # Add data rows (limit to first 1000 rows for performance)
        max_rows = min(1000, df.shape[0])
        for i in range(max_rows):
            values = df.iloc[i].to_list()
            # Format values for display
            formatted_values = []
            for val in values:
                if isinstance(val, (float, np.float64)):
                    formatted_values.append(f"{val:.4f}")
                else:
                    formatted_values.append(str(val))
            self.tree.insert('', tk.END, values=formatted_values)
        
        if max_rows < df.shape[0]:
            self.tree.insert('', tk.END, values=["..." for _ in range(len(df.columns))])
    
    def update_normal_columns(self, event=None):
        """Update column selection for normal distribution tab"""
        selected = self.normal_dataset_var.get()
        if not selected or selected not in self.datasets:
            return
        
        df = self.datasets[selected]
        # Only include numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.normal_column_combo['values'] = numeric_cols
        if numeric_cols:
            self.normal_column_var.set(numeric_cols[0])
    
    def update_poisson_columns(self, event=None):
        """Update column selection for poisson distribution tab"""
        selected = self.poisson_dataset_var.get()
        if not selected or selected not in self.datasets:
            return
        
        df = self.datasets[selected]
        # Only include integer or count-like columns for Poisson
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.poisson_column_combo['values'] = numeric_cols
        if numeric_cols:
            self.poisson_column_var.set(numeric_cols[0])
    
    def update_comparison_columns(self, dataset_num):
        """Update column selection for comparison tab"""
        if dataset_num == 1:
            selected = self.comp_dataset1_var.get()
            column_combo = self.comp_column1_combo
            column_var = self.comp_column1_var
        else:
            selected = self.comp_dataset2_var.get()
            column_combo = self.comp_column2_combo
            column_var = self.comp_column2_var
        
        if not selected or selected not in self.datasets:
            return
        
        df = self.datasets[selected]
        # Only include numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        column_combo['values'] = numeric_cols
        if numeric_cols:
            column_var.set(numeric_cols[0])
    
    def load_dataset(self):
        """Load a dataset from CSV file"""
        filetypes = [("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(title="Select a data file", filetypes=filetypes)
        
        if not filepath:
            return
        
        try:
            filename = os.path.basename(filepath)
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx'):
                df = pd.read_excel(filepath)
            else:
                messagebox.showerror("Error", "Unsupported file format")
                return
            
            # Add to datasets dictionary
            self.datasets[filename] = df
            self.recent_files.append(filepath)
            
            # Update combo boxes
            self.update_dataset_combos()
            
            self.status_var.set(f"Loaded dataset: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def export_results(self):
        """Export analysis results to file"""
        current_tab = self.notebook.tab(self.notebook.select(), "text")
        
        filetypes = [("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        filepath = filedialog.asksaveasfilename(title=f"Export {current_tab} Results", 
                                            defaultextension=".txt",
                                            filetypes=filetypes)
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'w') as f:
                if current_tab == "Normal Distribution":
                    f.write("NORMAL DISTRIBUTION ANALYSIS\n\n")
                    f.write("Dataset: " + self.normal_dataset_var.get() + "\n")
                    f.write("Column: " + self.normal_column_var.get() + "\n\n")
                    f.write("STATISTICS\n")
                    f.write(self.normal_stats_text.get("1.0", tk.END))
                    f.write("\nGOODNESS OF FIT\n")
                    f.write(self.poisson_gof_text.get("1.0", tk.END))
                
                elif current_tab == "Poisson Distribution":
                    f.write("POISSON DISTRIBUTION ANALYSIS\n\n")
                    f.write("Dataset: " + self.poisson_dataset_var.get() + "\n")
                    f.write("Column: " + self.poisson_column_var.get() + "\n\n")
                    f.write("STATISTICS\n")
                    f.write(self.poisson_stats_text.get("1.0", tk.END))
                    f.write("\nGOODNESS OF FIT\n")
                    f.write(self.poisson_gof_text.get("1.0", tk.END))
                
                elif current_tab == "Comparison":
                    f.write("DISTRIBUTION COMPARISON\n\n")
                    f.write("Dataset 1: " + self.comp_dataset1_var.get() + " - " + self.comp_column1_var.get() + "\n")
                    f.write("Dataset 2: " + self.comp_dataset2_var.get() + " - " + self.comp_column2_var.get() + "\n\n")
                    f.write(self.comp_stats_text.get("1.0", tk.END))
            
            self.status_var.set(f"Results exported to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def analyze_normal(self):
        """Analyze data for normal distribution fit"""
        dataset = self.normal_dataset_var.get()
        column = self.normal_column_var.get()
        
        if not dataset or not column:
            return
        
        # Use cached results if available
        cache_key = f"normal_{dataset}_{column}"
        if cache_key in self.analysis_cache:
            self.display_normal_results(self.analysis_cache[cache_key])
            return
        
        # Set status
        self.status_var.set("Analyzing normal distribution...")
        
        # Run analysis in a separate thread
        def run_analysis():
            try:
                df = self.datasets[dataset]
                data = df[column].dropna().values
                
                # Basic statistics
                mean = np.mean(data)
                std = np.std(data)
                median = np.median(data)
                skewness = stats.skew(data)
                kurtosis = stats.kurtosis(data)
                
                # Normal probability plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                # Histogram with normal fit
                hist, bins, _ = ax1.hist(data, bins=30, density=True, alpha=0.6, color='lightblue', edgecolor='black')
                x = np.linspace(min(data), max(data), 100)
                ax1.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2)
                ax1.set_title('Histogram with Normal Fit')
                ax1.set_xlabel(column)
                ax1.set_ylabel('Density')
                
                # QQ plot
                stats.probplot(data, dist="norm", plot=ax2)
                ax2.set_title('Q-Q Plot')
                
                fig.tight_layout()
                
                # Goodness of fit tests
                shapiro_test = stats.shapiro(data)
                ks_test = stats.kstest(data, 'norm', args=(mean, std))
                
                # Calculate confidence intervals
                n = len(data)
                ci_mean = stats.norm.interval(0.95, loc=mean, scale=std/np.sqrt(n))
                
                results = {
                    'data': data,
                    'mean': mean,
                    'std': std,
                    'median': median,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'shapiro_test': shapiro_test,
                    'ks_test': ks_test,
                    'ci_mean': ci_mean,
                    'fig': fig
                }
                
                                # Cache results
                self.analysis_cache[cache_key] = results
                
                # Update UI in main thread
                self.after(0, lambda: self.display_normal_results(results))
                self.after(0, lambda: self.status_var.set("Normal analysis completed"))
                
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
                self.after(0, lambda: self.status_var.set("Error in normal analysis"))
        
        # Start analysis thread
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def display_normal_results(self, results):
        """Display the results of normal distribution analysis"""
        # Clear previous figure
        for widget in self.normal_fig_frame.winfo_children():
            widget.destroy()
        
        # Embed matplotlib figure
        canvas = FigureCanvasTkAgg(results['fig'], master=self.normal_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update statistics text
        self.normal_stats_text.config(state=tk.NORMAL)
        self.normal_stats_text.delete(1.0, tk.END)
        
        stats_text = f"""Basic Statistics:
Mean: {results['mean']:.4f}
Standard Deviation: {results['std']:.4f}
Median: {results['median']:.4f}
Skewness: {results['skewness']:.4f}
Kurtosis: {results['kurtosis']:.4f}

95% Confidence Interval for Mean:
({results['ci_mean'][0]:.4f}, {results['ci_mean'][1]:.4f})
"""
        self.normal_stats_text.insert(tk.END, stats_text)
        self.normal_stats_text.config(state=tk.DISABLED)
        
        # Update goodness of fit text
        self.normal_gof_text.config(state=tk.NORMAL)
        self.normal_gof_text.delete(1.0, tk.END)
        
        gof_text = f"""Shapiro-Wilk Test:
Statistic: {results['shapiro_test'][0]:.4f}
p-value: {results['shapiro_test'][1]:.4f}

Kolmogorov-Smirnov Test:
Statistic: {results['ks_test'][0]:.4f}
p-value: {results['ks_test'][1]:.4f}

Interpretation:
- p-value > 0.05 suggests normal distribution
- p-value <= 0.05 suggests non-normal distribution
"""
        self.normal_gof_text.insert(tk.END, gof_text)
        self.normal_gof_text.config(state=tk.DISABLED)
    
    def analyze_poisson(self):
        """Analyze data for Poisson distribution fit"""
        dataset = self.poisson_dataset_var.get()
        column = self.poisson_column_var.get()
        time_interval = float(self.time_interval_var.get())
        
        if not dataset or not column or time_interval <= 0:
            return
        
        # Use cached results if available
        cache_key = f"poisson_{dataset}_{column}_{time_interval}"
        if cache_key in self.analysis_cache:
            self.display_poisson_results(self.analysis_cache[cache_key])
            return
        
        # Set status
        self.status_var.set("Analyzing Poisson distribution...")
        
        # Run analysis in a separate thread
        def run_analysis():
            try:
                df = self.datasets[dataset]
                data = df[column].dropna().values
                
                # Adjust for time interval if needed
                if time_interval != 1:
                    data = data / time_interval
                
                # Basic statistics
                mean = np.mean(data)
                var = np.var(data)
                dispersion = var / mean  # Dispersion index
                
                # Poisson probability mass function
                x = np.arange(0, int(np.max(data)) + 1)
                pmf = stats.poisson.pmf(x, mean)
                
                # Create figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                # Histogram with Poisson fit
                hist, bins, _ = ax1.hist(data, bins=30, density=True, alpha=0.6, color='lightgreen', edgecolor='black')
                ax1.plot(x, pmf, 'ro-', ms=5, label='Poisson PMF')
                ax1.vlines(x, 0, pmf, colors='r', lw=2, alpha=0.5)
                ax1.set_title('Histogram with Poisson Fit')
                ax1.set_xlabel(f'Counts per {time_interval} unit(s)')
                ax1.set_ylabel('Probability')
                ax1.legend()
                
                # Empirical vs theoretical CDF
                ecdf = np.arange(1, len(data)+1) / len(data)
                tcdf = stats.poisson.cdf(np.sort(data), mean)
                ax2.plot(np.sort(data), ecdf, 'b-', label='Empirical CDF')
                ax2.plot(np.sort(data), tcdf, 'r--', label='Poisson CDF')
                ax2.set_title('Empirical vs Theoretical CDF')
                ax2.set_xlabel(f'Counts per {time_interval} unit(s)')
                ax2.set_ylabel('Cumulative Probability')
                ax2.legend()
                
                fig.tight_layout()
                
                # Goodness of fit tests
                chi2_test = stats.chisquare(f_obs=np.histogram(data, bins=len(x))[0], 
                                          f_exp=len(data)*pmf)
                
                # Calculate confidence interval for lambda
                n = len(data)
                ci_lambda = stats.norm.interval(0.95, loc=mean, scale=np.sqrt(mean/n))
                
                results = {
                    'data': data,
                    'mean': mean,
                    'variance': var,
                    'dispersion': dispersion,
                    'chi2_test': chi2_test,
                    'ci_lambda': ci_lambda,
                    'time_interval': time_interval,
                    'fig': fig
                }
                
                # Cache results
                self.analysis_cache[cache_key] = results
                
                # Update UI in main thread
                self.after(0, lambda: self.display_poisson_results(results))
                self.after(0, lambda: self.status_var.set("Poisson analysis completed"))
                
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
                self.after(0, lambda: self.status_var.set("Error in Poisson analysis"))
        
        # Start analysis thread
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def display_poisson_results(self, results):
        """Display the results of Poisson distribution analysis"""
        # Clear previous figure
        for widget in self.poisson_fig_frame.winfo_children():
            widget.destroy()
        
        # Embed matplotlib figure
        canvas = FigureCanvasTkAgg(results['fig'], master=self.poisson_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update statistics text
        self.poisson_stats_text.config(state=tk.NORMAL)
        self.poisson_stats_text.delete(1.0, tk.END)
        
        stats_text = f"""Basic Statistics:
Mean (λ): {results['mean']:.4f}
Variance: {results['variance']:.4f}
Dispersion Index (Variance/Mean): {results['dispersion']:.4f}

95% Confidence Interval for λ:
({results['ci_lambda'][0]:.4f}, {results['ci_lambda'][1]:.4f})

Time Interval: {results['time_interval']}
"""
        self.poisson_stats_text.insert(tk.END, stats_text)
        self.poisson_stats_text.config(state=tk.DISABLED)
        
        # Update goodness of fit text
        self.poisson_gof_text.config(state=tk.NORMAL)
        self.poisson_gof_text.delete(1.0, tk.END)
        
        gof_text = f"""Chi-Square Goodness-of-Fit Test:
Statistic: {results['chi2_test'][0]:.4f}
p-value: {results['chi2_test'][1]:.4f}

Interpretation:
- Dispersion Index ≈ 1 suggests Poisson distribution
- p-value > 0.05 suggests Poisson distribution
- p-value <= 0.05 suggests non-Poisson distribution
"""
        self.poisson_gof_text.insert(tk.END, gof_text)
        self.poisson_gof_text.config(state=tk.DISABLED)
    
    def compare_distributions(self):
        """Compare two distributions"""
        dataset1 = self.comp_dataset1_var.get()
        column1 = self.comp_column1_var.get()
        dataset2 = self.comp_dataset2_var.get()
        column2 = self.comp_column2_var.get()
        
        if not dataset1 or not column1 or not dataset2 or not column2:
            return
        
        # Use cached results if available
        cache_key = f"compare_{dataset1}_{column1}_{dataset2}_{column2}"
        if cache_key in self.analysis_cache:
            self.display_comparison_results(self.analysis_cache[cache_key])
            return
        
        # Set status
        self.status_var.set("Comparing distributions...")
        
        # Run analysis in a separate thread
        def run_analysis():
            try:
                df1 = self.datasets[dataset1]
                data1 = df1[column1].dropna().values
                
                df2 = self.datasets[dataset2]
                data2 = df2[column2].dropna().values
                
                # Create figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                # Histograms
                ax1.hist(data1, bins=30, alpha=0.5, label=f"{dataset1}:{column1}")
                ax1.hist(data2, bins=30, alpha=0.5, label=f"{dataset2}:{column2}")
                ax1.set_title('Histograms Comparison')
                ax1.set_xlabel('Values')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                
                # ECDFs
                x1 = np.sort(data1)
                y1 = np.arange(1, len(data1)+1) / len(data1)
                x2 = np.sort(data2)
                y2 = np.arange(1, len(data2)+1) / len(data2)
                
                ax2.plot(x1, y1, label=f"{dataset1}:{column1}")
                ax2.plot(x2, y2, label=f"{dataset2}:{column2}")
                ax2.set_title('Empirical CDFs Comparison')
                ax2.set_xlabel('Values')
                ax2.set_ylabel('Cumulative Probability')
                ax2.legend()
                
                fig.tight_layout()
                
                # Statistical tests
                t_test = stats.ttest_ind(data1, data2, equal_var=False)
                ks_test = stats.ks_2samp(data1, data2)
                mwu_test = stats.mannwhitneyu(data1, data2)
                
                # Basic statistics
                stats1 = {
                    'mean': np.mean(data1),
                    'median': np.median(data1),
                    'std': np.std(data1),
                    'size': len(data1)
                }
                
                stats2 = {
                    'mean': np.mean(data2),
                    'median': np.median(data2),
                    'std': np.std(data2),
                    'size': len(data2)
                }
                
                results = {
                    'dataset1': f"{dataset1}:{column1}",
                    'dataset2': f"{dataset2}:{column2}",
                    'stats1': stats1,
                    'stats2': stats2,
                    't_test': t_test,
                    'ks_test': ks_test,
                    'mwu_test': mwu_test,
                    'fig': fig
                }
                
                # Cache results
                self.analysis_cache[cache_key] = results
                
                # Update UI in main thread
                self.after(0, lambda: self.display_comparison_results(results))
                self.after(0, lambda: self.status_var.set("Comparison completed"))
                
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Comparison failed: {str(e)}"))
                self.after(0, lambda: self.status_var.set("Error in comparison"))
        
        # Start analysis thread
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def display_comparison_results(self, results):
        """Display the results of distribution comparison"""
        # Clear previous figure
        for widget in self.comp_fig_frame.winfo_children():
            widget.destroy()
        
        # Embed matplotlib figure
        canvas = FigureCanvasTkAgg(results['fig'], master=self.comp_fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update statistics text
        self.comp_stats_text.config(state=tk.NORMAL)
        self.comp_stats_text.delete(1.0, tk.END)
        
        stats_text = f"""Dataset 1: {results['dataset1']}
Mean: {results['stats1']['mean']:.4f}
Median: {results['stats1']['median']:.4f}
Std Dev: {results['stats1']['std']:.4f}
Sample Size: {results['stats1']['size']}

Dataset 2: {results['dataset2']}
Mean: {results['stats2']['mean']:.4f}
Median: {results['stats2']['median']:.4f}
Std Dev: {results['stats2']['std']:.4f}
Sample Size: {results['stats2']['size']}

Statistical Tests:
Welch's t-test:
  Statistic: {results['t_test'][0]:.4f}
  p-value: {results['t_test'][1]:.4f}

Kolmogorov-Smirnov test:
  Statistic: {results['ks_test'][0]:.4f}
  p-value: {results['ks_test'][1]:.4f}

Mann-Whitney U test:
  Statistic: {results['mwu_test'][0]:.4f}
  p-value: {results['mwu_test'][1]:.4f}

Interpretation:
- p-value > 0.05 suggests similar distributions
- p-value <= 0.05 suggests different distributions
"""
        self.comp_stats_text.insert(tk.END, stats_text)
        self.comp_stats_text.config(state=tk.DISABLED)
    
    def show_help(self):
        """Show help information"""
        help_text = """Probability Distribution Analyzer Help

1. Load Data:
   - Click 'Load Dataset' to import CSV or Excel files
   - Default datasets are loaded automatically if found

2. Data View:
   - View and inspect loaded datasets
   - Basic statistics are shown for selected dataset

3. Normal Distribution Analysis:
   - Select dataset and numeric column
   - Click 'Analyze' to fit normal distribution
   - View histogram, Q-Q plot, and goodness-of-fit tests

4. Poisson Distribution Analysis:
   - Select dataset and count-based column
   - Adjust time interval if needed
   - Click 'Analyze' to fit Poisson distribution
   - View histogram, CDF, and goodness-of-fit tests

5. Comparison:
   - Select two datasets/columns to compare
   - Click 'Compare' to run statistical tests
   - View histograms, ECDFs, and test results

6. Export Results:
   - Export analysis results to text file

For more information, consult the documentation.
"""
        messagebox.showinfo("Help", help_text)  

if __name__ == "__main__":
    app = ProbabilityDistributionAnalyzer()
    app.mainloop()
