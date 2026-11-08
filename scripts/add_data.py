#!/usr/bin/env python3
"""
TrustedText Data CLI - Add training data with style
"""
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich import box
import inquirer

console = Console()

# data directory is at project root, not in scripts/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HUMAN_DIR = DATA_DIR / "human"
AI_DIR = DATA_DIR / "ai"


def clear_screen():
    """Clear the terminal screen"""
    console.clear()


def show_banner():
    """Display cool ASCII banner"""
    banner = Text()
    banner.append("╔════════════════════════════════════════╗\n", style="bold cyan")
    banner.append("║       TrustedText Data Collector       ║\n", style="bold cyan")
    banner.append("╚════════════════════════════════════════╝", style="bold cyan")
    console.print(banner)
    console.print()


def get_dataset_stats() -> dict:
    """Get current dataset statistics"""
    def count_files(directory: Path) -> int:
        if not directory.exists():
            return 0
        return len(list(directory.glob("*.txt")))

    human_files = count_files(HUMAN_DIR)
    ai_files = count_files(AI_DIR)

    return {
        "human_files": human_files,
        "human_samples": human_files,
        "ai_files": ai_files,
        "ai_samples": ai_files,
    }


def show_stats():
    """Display dataset statistics in a beautiful table"""
    stats = get_dataset_stats()
    
    table = Table(title="Current Dataset", box=box.ROUNDED, title_style="bold magenta")
    table.add_column("Category", style="cyan", justify="left")
    table.add_column("Files", style="yellow", justify="right")
    table.add_column("Samples", style="green", justify="right")
    
    table.add_row("🧑 Human", str(stats["human_files"]), str(stats["human_samples"]))
    table.add_row("🤖 AI", str(stats["ai_files"]), str(stats["ai_samples"]))
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{stats['human_files'] + stats['ai_files']}[/bold]",
        f"[bold]{stats['human_samples'] + stats['ai_samples']}[/bold]"
    )
    
    console.print(table)
    console.print()


def get_next_filename(directory: Path, prefix: str) -> str:
    """Generate next available filename"""
    existing = list(directory.glob(f"{prefix}*.txt"))
    if not existing:
        return f"{prefix}001.txt"
    
    # Extract numbers and find the highest
    numbers = []
    for f in existing:
        name = f.stem
        num_part = name.replace(prefix, "")
        if num_part.isdigit():
            numbers.append(int(num_part))
    
    if numbers:
        next_num = max(numbers) + 1
        return f"{prefix}{next_num:03d}.txt"
    return f"{prefix}001.txt"


def add_text_sample():
    """Add a new text sample to the dataset"""
    console.print()
    console.print(Panel.fit(
        "[yellow]Add New Training Sample[/yellow]",
        border_style="yellow"
    ))
    console.print()
    
    # Ask for category
    questions = [
        inquirer.List(
            'category',
            message="What type of text is this?",
            choices=['🧑 Human-written', '🤖 AI-generated'],
        ),
    ]
    
    try:
        answers = inquirer.prompt(questions)
        if not answers:
            return False
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Cancelled[/yellow]")
        return False
    
    is_human = answers['category'].startswith('🧑')
    category_name = "Human" if is_human else "AI"
    target_dir = HUMAN_DIR if is_human else AI_DIR
    
    # Get filename prefix
    console.print()
    default_prefix = "custom"
    prefix = Prompt.ask(
        f"[cyan]Filename prefix[/cyan] (e.g., 'news', 'blog', 'chat')",
        default=default_prefix
    )
    
    # Clean prefix
    prefix = "".join(c for c in prefix if c.isalnum() or c in "-_").lower()
    if not prefix:
        prefix = default_prefix
    
    filename = get_next_filename(target_dir, prefix)
    
    # Get the text content
    console.print()
    console.print("[cyan]Enter or paste the text (press Ctrl+D or Ctrl+Z when done):[/cyan]")
    console.print("[dim]You can paste multiple lines. Empty lines will be preserved.[/dim]")
    console.print()
    
    try:
        lines = []
        while True:
            try:
                line = input()
                lines.append(line)
            except EOFError:
                break
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        return False
    
    text_content = "\n".join(lines).strip()
    
    if not text_content:
        console.print("[red]No text provided. Sample not saved.[/red]")
        return False
    
    # Preview
    console.print()
    preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
    console.print(Panel(
        preview,
        title=f"[green]Preview[/green]",
        border_style="green"
    ))
    
    # Confirm
    console.print()
    console.print(f"[cyan]Category:[/cyan] {category_name}")
    console.print(f"[cyan]Filename:[/cyan] {filename}")
    console.print(f"[cyan]Characters:[/cyan] {len(text_content)}")
    console.print()
    
    if not Confirm.ask("[yellow]Save this sample?[/yellow]", default=True):
        console.print("[yellow]Sample not saved[/yellow]")
        return False
    
    # Save the file
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / filename
    filepath.write_text(text_content, encoding="utf-8")
    
    console.print()
    console.print(Panel.fit(
        f"[green]✓ Saved to {filepath.relative_to(Path.cwd())}[/green]",
        border_style="green"
    ))
    
    return True


def main_menu():
    """Display main menu and handle user choice"""
    while True:
        clear_screen()
        show_banner()
        show_stats()
        
        questions = [
            inquirer.List(
                'action',
                message="What would you like to do?",
                choices=[
                    '➕ Add new sample',
                    '📊 Refresh stats',
                    '🚪 Exit'
                ],
            ),
        ]
        
        try:
            answers = inquirer.prompt(questions)
            if not answers:
                break
                
            action = answers['action']
            
            if action.startswith('➕'):
                add_text_sample()
                console.print()
                Prompt.ask("[dim]Press Enter to continue[/dim]", default="")
            elif action.startswith('📊'):
                continue
            elif action.startswith('🚪'):
                break
                
        except (KeyboardInterrupt, EOFError):
            break
    
    console.print()
    console.print("[cyan]Thanks for contributing to TrustedText![/cyan] 🎉")
    console.print()


if __name__ == "__main__":
    try:
        main_menu()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
