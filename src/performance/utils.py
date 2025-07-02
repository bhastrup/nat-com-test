
import os, sys


# Global variable to control print redirection
REDIRECT_PRINT = False

def no_print_wrapper(func: callable, *args, **kwargs):
    """Wrapper to suppress stdout and stderr during the execution of a function"""
    if not REDIRECT_PRINT:
        # If REDIRECT_PRINT is False, just call the function normally
        return func(*args, **kwargs)

    # Save the original stdout and stderr
    original_stdout = os.dup(sys.stdout.fileno())
    original_stderr = os.dup(sys.stderr.fileno())

    # Open /dev/null to redirect the outputs
    with open(os.devnull, 'w') as fnull:
        # Redirect stdout and stderr to /dev/null
        os.dup2(fnull.fileno(), sys.stdout.fileno())
        os.dup2(fnull.fileno(), sys.stderr.fileno())

        try:
            # Execute the function with args and kwargs
            result = func(*args, **kwargs)
        finally:
            # Restore stdout and stderr back to their original file descriptors
            os.dup2(original_stdout, sys.stdout.fileno())
            os.dup2(original_stderr, sys.stderr.fileno())

            # Close the duplicated file descriptors
            os.close(original_stdout)
            os.close(original_stderr)

    return result
