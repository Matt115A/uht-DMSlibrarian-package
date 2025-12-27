#!/usr/bin/env python3
"""
Gradio GUI for uht-DMSlibrarian
Beautiful web interface for running the complete pipeline workflow.
"""

import os
import sys
import argparse
import threading
import time
import queue
import io
from pathlib import Path
from typing import Optional, Tuple, List
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    import gradio as gr
except ImportError:
    print("ERROR: gradio is not installed. Please install it with: pip install gradio")
    sys.exit(1)

# Import CLI functions
from .cli import (
    run_full_pipeline, 
    count_fasta_sequences, 
    count_files_in_directory,
    get_file_size_mb
)
from . import ngs_count
from . import fitness_analysis


# Global state for progress tracking
metrics_history = {
    'pipeline': {'time': [], 'umis': [], 'clusters': [], 'consensus': [], 'variants': []},
    'ngs': {'time': [], 'pools_processed': [], 'matches': []},
    'fitness': {'time': [], 'variants_processed': []}
}


def create_metrics_plot(metrics_type: str = 'pipeline'):
    """Create a real-time metrics plot."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(f'{metrics_type.capitalize()} Metrics', fontsize=14, fontweight='bold')
    
    history = metrics_history.get(metrics_type, {})
    
    if metrics_type == 'pipeline' and history.get('time') and len(history['time']) > 0:
        times = np.array(history['time'])
        if len(times) > 0:
            axes[0].plot(times, history.get('umis', [0]*len(times)), label='UMIs Extracted', marker='o')
            axes[0].plot(times, history.get('clusters', [0]*len(times)), label='Clusters', marker='s')
            axes[0].plot(times, history.get('consensus', [0]*len(times)), label='Consensus', marker='^')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Processing Progress')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_pipeline_with_progress(
    input_fastq, probe_file, reference_file, output_dir,
    umi_len, umi_loc, min_probe_score, clustering_method,
    identity, aln_thresh, size_thresh, max_reads, max_workers, report_file,
    progress=gr.Progress()
):
    """Run pipeline with progress tracking."""
    try:
        # Get file paths from Gradio file objects
        input_path = input_fastq.name if hasattr(input_fastq, 'name') else input_fastq
        probe_path = probe_file.name if hasattr(probe_file, 'name') else probe_file
        reference_path = reference_file.name if hasattr(reference_file, 'name') else reference_file
        
        # Validate inputs
        if not all([input_path, probe_path, reference_path, output_dir]):
            return "Error: All required fields must be filled", None, None, create_metrics_plot('pipeline')
        
        # Create args namespace
        args = argparse.Namespace(
            input=input_path,
            probe=probe_path,
            reference=reference_path,
            output_dir=output_dir,
            umi_len=int(umi_len) if umi_len else 52,
            umi_loc=umi_loc if umi_loc else 'up',
            min_probe_score=int(min_probe_score) if min_probe_score else 15,
            fast=(clustering_method == 'fast'),
            slow=(clustering_method == 'slow'),
            identity=float(identity) if identity else 0.90,
            aln_thresh=float(aln_thresh) if aln_thresh else 0.47,
            size_thresh=int(size_thresh) if size_thresh else 10,
            max_reads=int(max_reads) if max_reads else 20,
            max_workers=int(max_workers) if max_workers else 4,
            report=report_file if report_file else None
        )
        
        # Capture output
        output_capture = io.StringIO()
        with redirect_stdout(output_capture), redirect_stderr(output_capture):
            success = run_full_pipeline(args)
        
        output_text = output_capture.getvalue()
        
        if success:
            # Collect output files
            output_files = []
            base_dir = Path(output_dir)
            if base_dir.exists():
                output_files.extend([
                    str(f) for f in base_dir.rglob("*.fasta") if f.is_file()
                ])
                output_files.extend([
                    str(f) for f in base_dir.rglob("*.vcf") if f.is_file()
                ])
                output_files.extend([
                    str(f) for f in base_dir.rglob("*.csv") if f.is_file()
                ])
                if report_file and os.path.exists(report_file):
                    output_files.append(report_file)
            
            status = "Pipeline completed successfully!"
            return status, output_text, output_files, create_metrics_plot('pipeline')
        else:
            status = "Pipeline failed. Check progress output for details."
            return status, output_text, None, create_metrics_plot('pipeline')
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        import traceback
        traceback_text = traceback.format_exc()
        return error_msg, traceback_text, None, create_metrics_plot('pipeline')


def run_ngs_with_progress(
    pools_dir, consensus_dir, variants_dir, ngs_probe_file, ngs_reference_file,
    ngs_output, ngs_umi_len, ngs_umi_loc, left_ignore, right_ignore,
    progress=gr.Progress()
):
    """Run NGS counting with progress tracking."""
    try:
        probe_path = ngs_probe_file.name if hasattr(ngs_probe_file, 'name') else ngs_probe_file
        reference_path = ngs_reference_file.name if hasattr(ngs_reference_file, 'name') else ngs_reference_file
        
        if not all([pools_dir, consensus_dir, variants_dir, probe_path, reference_path, ngs_output]):
            return "Error: All required fields must be filled", None, None, create_metrics_plot('ngs')
        
        output_capture = io.StringIO()
        with redirect_stdout(output_capture), redirect_stderr(output_capture):
            success = ngs_count.run_ngs_count(
                pools_dir,
                consensus_dir,
                variants_dir,
                probe_path,
                int(ngs_umi_len) if ngs_umi_len else 52,
                ngs_umi_loc if ngs_umi_loc else 'up',
                ngs_output,
                reference_path,
                int(left_ignore) if left_ignore else 22,
                int(right_ignore) if right_ignore else 24
            )
        
        output_text = output_capture.getvalue()
        
        if success:
            output_files = [ngs_output] if os.path.exists(ngs_output) else []
            output_dir = Path(ngs_output).parent
            if output_dir.exists():
                output_files.extend([str(f) for f in output_dir.glob("*pool*.csv") if f.is_file()])
            
            status = "NGS counting completed successfully!"
            return status, output_text, output_files, create_metrics_plot('ngs')
        else:
            status = "NGS counting failed. Check progress output for details."
            return status, output_text, None, create_metrics_plot('ngs')
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        import traceback
        traceback_text = traceback.format_exc()
        return error_msg, traceback_text, None, create_metrics_plot('ngs')


def run_fitness_with_progress(
    fitness_input_csv, fitness_output_dir, input_pools, output_pools,
    min_input, aa_filter,
    progress=gr.Progress()
):
    """Run fitness analysis with progress tracking."""
    try:
        csv_path = fitness_input_csv.name if hasattr(fitness_input_csv, 'name') else fitness_input_csv
        
        if not all([csv_path, fitness_output_dir, input_pools, output_pools]):
            return "❌ Error: All required fields must be filled", None, None, create_metrics_plot('fitness')
        
        # Parse pool lists
        input_pool_list = [p.strip() for p in input_pools.split() if p.strip()]
        output_pool_list = [p.strip() for p in output_pools.split() if p.strip()]
        
        if len(input_pool_list) != len(output_pool_list):
            return "❌ Error: Number of input pools must match output pools", None, None, create_metrics_plot('fitness')
        
        output_capture = io.StringIO()
        with redirect_stdout(output_capture), redirect_stderr(output_capture):
            success = fitness_analysis.run_fitness_analysis(
                csv_path,
                fitness_output_dir,
                input_pool_list,
                output_pool_list,
                int(min_input) if min_input else 10,
                aa_filter if aa_filter else None
            )
        
        output_text = output_capture.getvalue()
        
        if success:
            output_files = []
            output_dir = Path(fitness_output_dir)
            if output_dir.exists():
                output_files.extend([str(f) for f in output_dir.rglob("*.csv") if f.is_file()])
                output_files.extend([str(f) for f in output_dir.rglob("*.png") if f.is_file()])
            
            status = "Fitness analysis completed successfully!"
            return status, output_text, output_files, create_metrics_plot('fitness')
        else:
            status = "Fitness analysis failed. Check progress output for details."
            return status, output_text, None, create_metrics_plot('fitness')
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        import traceback
        traceback_text = traceback.format_exc()
        return error_msg, traceback_text, None, create_metrics_plot('fitness')


def auto_populate_ngs_from_pipeline(pipeline_output_dir):
    """Auto-populate NGS tab fields from pipeline output."""
    if not pipeline_output_dir or not os.path.exists(pipeline_output_dir):
        return gr.update(value=""), gr.update(value=""), gr.update(), gr.update(), gr.update(value=52), gr.update(value="up"), gr.update(value=22), gr.update(value=24)
    
    base = Path(pipeline_output_dir)
    consensus_dir = str(base / "consensus") if (base / "consensus").exists() else ""
    variants_dir = str(base / "variants") if (base / "variants").exists() else ""
    
    return (
        gr.update(value=consensus_dir),
        gr.update(value=variants_dir),
        gr.update(),  # Probe file - user must select
        gr.update(),  # Reference file - user must select
        gr.update(value=52),
        gr.update(value="up"),
        gr.update(value=22),
        gr.update(value=24)
    )


def auto_populate_fitness_from_ngs(ngs_output_file):
    """Auto-populate fitness tab from NGS output."""
    if not ngs_output_file:
        return gr.update(), gr.update(value=""), gr.update(value=""), gr.update(value="")
    
    # Handle both file list and single file
    if isinstance(ngs_output_file, list) and len(ngs_output_file) > 0:
        ngs_path = ngs_output_file[0].name if hasattr(ngs_output_file[0], 'name') else ngs_output_file[0]
    else:
        ngs_path = ngs_output_file.name if hasattr(ngs_output_file, 'name') else ngs_output_file
    
    if not ngs_path or not os.path.exists(ngs_path):
        return gr.update(), gr.update(value=""), gr.update(value=""), gr.update(value="")
    
    output_dir = Path(ngs_path).parent
    merged_file = output_dir / "merged_on_nonsyn_counts.csv"
    
    if merged_file.exists():
        try:
            df = pd.read_csv(merged_file)
            metadata_cols = ['AA_MUTATIONS', 'CONSENSUS_IDS', 'NUC_MUTATIONS']
            pool_cols = [col for col in df.columns if col not in metadata_cols]
            
            mid = len(pool_cols) // 2
            input_suggested = " ".join(pool_cols[:mid]) if mid > 0 else ""
            output_suggested = " ".join(pool_cols[mid:]) if mid < len(pool_cols) else ""
            
            fitness_output_dir = str(output_dir / "fitness_results")
            # Return file path for file component
            return gr.update(value=str(merged_file)), gr.update(value=fitness_output_dir), gr.update(value=input_suggested), gr.update(value=output_suggested)
        except Exception as e:
            fitness_output_dir = str(output_dir / "fitness_results")
            return gr.update(value=str(merged_file)), gr.update(value=fitness_output_dir), gr.update(value=""), gr.update(value="")
    
    return gr.update(), gr.update(value=""), gr.update(value=""), gr.update(value="")


def create_interface():
    """Create the main Gradio interface."""
    with gr.Blocks(title="uht-DMSlibrarian Pipeline GUI", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🧬 uht-DMSlibrarian Pipeline GUI
            
            Complete pipeline for UMIC-seq PacBio data processing, NGS pool counting, and fitness analysis.
            
            **Workflow**: Run Pipeline → NGS Count → Fitness Analysis (in that order)
            """
        )
        
        with gr.Tabs() as tabs:
            # Pipeline Tab
            with gr.Tab("Pipeline"):
                gr.Markdown("##Complete Pipeline\nRun the full UMIC-seq PacBio pipeline from raw reads to variant analysis.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        pipeline_input_fastq = gr.File(
                            label="Input FASTQ File * (Raw PacBio reads, can be .gz compressed)"
                        )
                        pipeline_probe_file = gr.File(
                            label="Probe FASTA File * (Probe sequence ~50bp adjacent to UMI)",
                            file_types=[".fasta", ".fa"]
                        )
                        pipeline_reference_file = gr.File(
                            label="Reference FASTA File * (Reference gene sequence for variant calling)",
                            file_types=[".fasta", ".fa"]
                        )
                        pipeline_output_dir = gr.Textbox(
                            label="Output Directory *",
                            placeholder="/path/to/output",
                            info="Directory where all results will be written"
                        )
                
                with gr.Accordion("UMI Extraction Parameters", open=False):
                    with gr.Row():
                        pipeline_umi_len = gr.Number(
                            label="UMI Length",
                            value=52,
                            minimum=1,
                            maximum=200,
                            precision=0,
                            info="Length of UMI in base pairs (default: 52)"
                        )
                        pipeline_umi_loc = gr.Dropdown(
                            label="UMI Location",
                            choices=["up", "down"],
                            value="up",
                            info="Location of UMI relative to probe: 'up' (upstream/5') or 'down' (downstream/3')"
                        )
                        pipeline_min_probe_score = gr.Number(
                            label="Min Probe Score",
                            value=15,
                            minimum=0,
                            precision=0,
                            info="Minimum alignment score for probe matching. Perfect match = probe length."
                        )
                
                with gr.Accordion("Clustering Parameters", open=False):
                    with gr.Row():
                        pipeline_clustering_method = gr.Radio(
                            label="Clustering Method",
                            choices=["fast", "slow"],
                            value="fast",
                            info="Fast uses CD-HIT (recommended, 1000x faster), Slow uses alignment-based clustering"
                        )
                        pipeline_identity = gr.Slider(
                            label="Identity Threshold (Fast)",
                            minimum=0.5,
                            maximum=1.0,
                            value=0.90,
                            step=0.01,
                            info="Sequence identity for fast clustering (0.90 = 90% identity, allows ~10% mismatch)"
                        )
                        pipeline_aln_thresh = gr.Slider(
                            label="Alignment Threshold (Slow)",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.47,
                            step=0.01,
                            visible=False,
                            info="Alignment score threshold for slow clustering (legacy method)"
                        )
                        pipeline_size_thresh = gr.Number(
                            label="Size Threshold",
                            value=10,
                            minimum=1,
                            precision=0,
                            info="Minimum reads per cluster. Lower = more sensitive, Higher = more conservative"
                        )
                
                with gr.Accordion("Consensus & Variant Parameters", open=False):
                    with gr.Row():
                        pipeline_max_reads = gr.Number(
                            label="Max Reads per Consensus",
                            value=20,
                            minimum=1,
                            precision=0,
                            info="Maximum reads used for consensus generation. More reads = better quality but slower"
                        )
                        pipeline_max_workers = gr.Number(
                            label="Max Workers",
                            value=4,
                            minimum=1,
                            precision=0,
                            info="Number of parallel workers. Increase for faster processing if you have more CPU cores"
                        )
                
                with gr.Accordion("Output Options", open=False):
                    pipeline_report_file = gr.Textbox(
                        label="Report File (Optional)",
                        placeholder="/path/to/pipeline_report.txt",
                        info="Generate a summary report with execution statistics"
                    )
                
                pipeline_clustering_method.change(
                    fn=lambda m: gr.update(visible=(m == "slow")),
                    inputs=pipeline_clustering_method,
                    outputs=pipeline_aln_thresh
                )
                
                pipeline_run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")
                
                with gr.Row():
                    pipeline_progress_text = gr.Textbox(
                        label="Progress",
                        lines=10,
                        max_lines=20,
                        interactive=False,
                        container=True
                    )
                    pipeline_metrics_plot = gr.Plot(label="Real-time Metrics")
                
                pipeline_status = gr.Textbox(label="Status", interactive=False)
                pipeline_output_files = gr.File(label="Output Files", file_count="multiple")
                
                pipeline_run_btn.click(
                    fn=run_pipeline_with_progress,
                    inputs=[
                        pipeline_input_fastq, pipeline_probe_file,
                        pipeline_reference_file, pipeline_output_dir,
                        pipeline_umi_len, pipeline_umi_loc,
                        pipeline_min_probe_score, pipeline_clustering_method,
                        pipeline_identity, pipeline_aln_thresh,
                        pipeline_size_thresh, pipeline_max_reads,
                        pipeline_max_workers, pipeline_report_file
                    ],
                    outputs=[
                        pipeline_status, pipeline_progress_text,
                        pipeline_output_files, pipeline_metrics_plot
                    ]
                )
            
            # NGS Count Tab
            with gr.Tab("NGS Count"):
                gr.Markdown("## NGS Pool Counting\nCount Illumina reads per variant via UMI matching to consensus sequences.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        ngs_pools_dir = gr.Textbox(
                            label="Pools Directory *",
                            placeholder="/path/to/NGS_data",
                            info="Directory containing per-pool folders with R1/R2 FASTQ files"
                        )
                        ngs_consensus_dir = gr.Textbox(
                            label="Consensus Directory *",
                            placeholder="/path/to/consensus",
                            info="Consensus sequences directory (from Pipeline step)"
                        )
                        ngs_variants_dir = gr.Textbox(
                            label="Variants Directory *",
                            placeholder="/path/to/variants",
                            info="Variants directory with per-consensus VCFs (from Pipeline step)"
                        )
                        ngs_probe_file = gr.File(
                            label="Probe FASTA File * (Same probe file used for Pipeline UMI extraction)",
                            file_types=[".fasta", ".fa"]
                        )
                        ngs_reference_file = gr.File(
                            label="Reference FASTA File * (Reference sequence for amino acid mapping)",
                            file_types=[".fasta", ".fa"]
                        )
                        ngs_output = gr.Textbox(
                            label="Output CSV File *",
                            placeholder="/path/to/pool_variant_counts.csv",
                            info="Output file for pool variant counts"
                        )
                
                with gr.Accordion("UMI Parameters", open=False):
                    with gr.Row():
                        ngs_umi_len = gr.Number(
                            label="UMI Length",
                            value=52,
                            minimum=1,
                            maximum=200,
                            precision=0,
                            info="Length of UMI (should match Pipeline setting)"
                        )
                        ngs_umi_loc = gr.Dropdown(
                            label="UMI Location",
                            choices=["up", "down"],
                            value="up",
                            info="UMI location relative to probe (should match Pipeline setting)"
                        )
                
                with gr.Accordion("Read Trimming Parameters", open=False):
                    with gr.Row():
                        ngs_left_ignore = gr.Number(
                            label="Left Ignore Bases",
                            value=22,
                            minimum=0,
                            precision=0,
                            info="Bases to ignore from start of assembled read (default: 22)"
                        )
                        ngs_right_ignore = gr.Number(
                            label="Right Ignore Bases",
                            value=24,
                            minimum=0,
                            precision=0,
                            info="Bases to ignore from end of assembled read (default: 24)"
                        )
                
                ngs_auto_populate_btn = gr.Button("Auto-populate from Pipeline", variant="secondary")
                ngs_run_btn = gr.Button("Run NGS Counting", variant="primary", size="lg")
                
                with gr.Row():
                    ngs_progress_text = gr.Textbox(
                        label="Progress",
                        lines=10,
                        max_lines=20,
                        interactive=False
                    )
                    ngs_metrics_plot = gr.Plot(label="Real-time Metrics")
                
                ngs_status = gr.Textbox(label="Status", interactive=False)
                ngs_output_files = gr.File(label="Output Files", file_count="multiple")
                
                ngs_auto_populate_btn.click(
                    fn=auto_populate_ngs_from_pipeline,
                    inputs=pipeline_output_dir,
                    outputs=[
                        ngs_consensus_dir, ngs_variants_dir,
                        ngs_probe_file, ngs_reference_file,
                        ngs_umi_len, ngs_umi_loc,
                        ngs_left_ignore, ngs_right_ignore
                    ]
                )
                ngs_run_btn.click(
                    fn=run_ngs_with_progress,
                    inputs=[
                        ngs_pools_dir, ngs_consensus_dir,
                        ngs_variants_dir, ngs_probe_file,
                        ngs_reference_file, ngs_output,
                        ngs_umi_len, ngs_umi_loc,
                        ngs_left_ignore, ngs_right_ignore
                    ],
                    outputs=[
                        ngs_status, ngs_progress_text,
                        ngs_output_files, ngs_metrics_plot
                    ]
                )
            
            # Fitness Analysis Tab
            with gr.Tab("Fitness Analysis"):
                gr.Markdown("## Fitness Analysis\nCalculate fitness from input/output pool comparisons with comprehensive visualizations.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        fitness_input_csv = gr.File(
                            label="Input CSV File * (merged_on_nonsyn_counts.csv from NGS Count step)",
                            file_types=[".csv"]
                        )
                        fitness_output_dir = gr.Textbox(
                            label="Output Directory *",
                            placeholder="/path/to/fitness_results",
                            info="Directory to save plots and processed data"
                        )
                
                with gr.Accordion("Pool Configuration", open=True):
                    gr.Markdown("**Input/Output Pool Pairs**: Enter pool names that represent input and output conditions.")
                    
                    with gr.Row():
                        fitness_input_pools = gr.Textbox(
                            label="Input Pool Names *",
                            placeholder="pool1 pool2",
                            info="Space-separated list of input pool names (e.g., 'pool1 pool2')"
                        )
                        fitness_output_pools = gr.Textbox(
                            label="Output Pool Names *",
                            placeholder="pool3 pool4",
                            info="Space-separated list of output pool names, paired with inputs (e.g., 'pool3 pool4')"
                        )
                
                with gr.Accordion("Filtering Parameters", open=False):
                    with gr.Row():
                        fitness_min_input = gr.Number(
                            label="Min Input Threshold",
                            value=10,
                            minimum=1,
                            precision=0,
                            info="Minimum count threshold in input pools. Variants below this are filtered out."
                        )
                        fitness_aa_filter = gr.Textbox(
                            label="Amino Acid Filter (Optional)",
                            placeholder="S",
                            info="Filter mutability plot to specific amino acid (e.g., 'S' for serine, 'P' for proline, '*' for stop codons)"
                        )
                
                fitness_auto_populate_btn = gr.Button("Auto-populate from NGS Count", variant="secondary")
                fitness_run_btn = gr.Button("Run Fitness Analysis", variant="primary", size="lg")
                
                with gr.Row():
                    fitness_progress_text = gr.Textbox(
                        label="Progress",
                        lines=10,
                        max_lines=20,
                        interactive=False
                    )
                    fitness_metrics_plot = gr.Plot(label="Real-time Metrics")
                
                fitness_status = gr.Textbox(label="Status", interactive=False)
                fitness_output_files = gr.File(label="Output Files", file_count="multiple")
                
                fitness_auto_populate_btn.click(
                    fn=auto_populate_fitness_from_ngs,
                    inputs=ngs_output,
                    outputs=[
                        fitness_input_csv, fitness_output_dir,
                        fitness_input_pools, fitness_output_pools
                    ]
                )
                fitness_run_btn.click(
                    fn=run_fitness_with_progress,
                    inputs=[
                        fitness_input_csv, fitness_output_dir,
                        fitness_input_pools, fitness_output_pools,
                        fitness_min_input, fitness_aa_filter
                    ],
                    outputs=[
                        fitness_status, fitness_progress_text,
                        fitness_output_files, fitness_metrics_plot
                    ]
                )
    
    return app


def launch_gui(server_name="127.0.0.1", server_port=7860, share=False):
    """Launch the Gradio GUI."""
    app = create_interface()
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True
    )


if __name__ == "__main__":
    import argparse as ap
    parser = ap.ArgumentParser(description="Launch uht-DMSlibrarian GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()
    
    launch_gui(server_name=args.host, server_port=args.port, share=args.share)
