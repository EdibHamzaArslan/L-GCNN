
import sys


yellow = "\x1b[33;20m"
green = "\x1b[32m"
red = "\x1b[31;20m"
reset = "\x1b[0m"

class Debug:
    
    @staticmethod
    def print(message, input):
        print(f"{red+message+reset} {input.shape}")
    
    @staticmethod
    def end():
        print(f"{yellow}Program is terminated!{reset}")
        sys.exit(1)
