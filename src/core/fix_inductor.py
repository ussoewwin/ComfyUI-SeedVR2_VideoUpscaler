import os
import inspect
import textwrap

def _fix_inductor_windows_encoding() -> None:
    """
    Monkey-patch torch._inductor.cpp_builder._run_compile_cmd to fix UnicodeDecodeError on Windows.
    This replaces .decode("utf-8") with .decode("utf-8", errors="replace").
    This allows compilation errors (which may contain Shift-JIS text) to be displayed instead of crashing.
    """
    if os.name != 'nt':
        return

    try:
        # Import the target module
        # Note: We use importlib to be safe, but direct import is fine inside function
        import torch._inductor.cpp_builder
        
        # Check if already patched (optional, but good practice if we set a flag)
        # For now, just apply it.
        
        target_func = torch._inductor.cpp_builder._run_compile_cmd
        
        # Get source code
        try:
            source = inspect.getsource(target_func)
        except OSError:
            # Source not available (compiled .pyc only etc.)
            return
            
        source = textwrap.dedent(source)
        
        # The exact line causing issues in the traceback: output = e.stdout.decode("utf-8")
        # We look for .decode("utf-8") and replace it
        old_code = 'e.stdout.decode("utf-8")'
        new_code = 'e.stdout.decode("utf-8", errors="replace")'
        
        if old_code in source:
            # Perform replacement
            new_source = source.replace(old_code, new_code)
            
            # Execute in the module's namespace to resolve globals correctly
            module = torch._inductor.cpp_builder
            
            # Capture the new function definition
            local_scope = {}
            # We pass module.__dict__ as globals so it can find imports like subprocess
            exec(new_source, module.__dict__, local_scope)
            
            # Replace the function in the module
            if '_run_compile_cmd' in local_scope:
                module._run_compile_cmd = local_scope['_run_compile_cmd']
                # print("[SeedVR2] Successfully patched torch.inductor for Windows encoding support")
            
    except Exception as e:
        # Fail silently or log to console, don't crash app
        print(f"[SeedVR2] Warning: Could not patch torch.inductor: {e}")

