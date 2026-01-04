#!/usr/bin/env python3
"""
Unified CLI entry point for pyCBS.

Supports the following commands:
  - pycbs -input: Process input files
  - pycbs -opt: Run optimization
"""

import argparse
import sys
from pathlib import Path


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog='pycbs',
        description='pyCBS - A Python-based Conflict-Based Search solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  pycbs -input config.yaml
  pycbs -opt --output results.json
        '''
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Input command
    input_parser = subparsers.add_parser(
        '-input',
        help='Process input files for pyCBS'
    )
    input_parser.add_argument(
        'config',
        type=str,
        help='Path to input configuration file'
    )
    input_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    input_parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path for processed input'
    )
    
    # Optimization command
    opt_parser = subparsers.add_parser(
        '-opt',
        help='Run optimization algorithm'
    )
    opt_parser.add_argument(
        '-i', '--input',
        type=str,
        required=False,
        help='Path to input file or configuration'
    )
    opt_parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path for optimization results'
    )
    opt_parser.add_argument(
        '--timeout',
        type=float,
        default=300.0,
        help='Timeout in seconds for optimization (default: 300.0)'
    )
    opt_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    opt_parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    return parser


def handle_input_command(args):
    """Handle the -input command."""
    if args.verbose:
        print(f"Processing input from: {args.config}")
    
    # Validate input file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Input file '{args.config}' not found", file=sys.stderr)
        return 1
    
    try:
        # TODO: Implement input processing logic
        if args.verbose:
            print(f"Successfully processed: {args.config}")
        
        if args.output:
            print(f"Output would be written to: {args.output}")
            if args.verbose:
                print(f"Output file: {args.output}")
        
        return 0
    except Exception as e:
        print(f"Error processing input: {e}", file=sys.stderr)
        return 1


def handle_opt_command(args):
    """Handle the -opt command."""
    if args.verbose:
        print("Starting optimization algorithm")
    
    try:
        if args.input and args.verbose:
            print(f"Using input: {args.input}")
        
        if args.seed is not None and args.verbose:
            print(f"Random seed: {args.seed}")
        
        print(f"Optimization timeout: {args.timeout} seconds")
        
        # TODO: Implement optimization logic
        if args.verbose:
            print("Optimization completed successfully")
        
        if args.output:
            print(f"Results would be written to: {args.output}")
            if args.verbose:
                print(f"Output file: {args.output}")
        
        return 0
    except Exception as e:
        print(f"Error running optimization: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle no command provided
    if args.command is None:
        parser.print_help()
        return 0
    
    # Route to appropriate command handler
    if args.command == '-input':
        return handle_input_command(args)
    elif args.command == '-opt':
        return handle_opt_command(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
