import sys
import linecache



def set_trace_to_file(filename, function, *args, **kwargs):
    with open(filename, "w") as file:
        def trace_function(frame, event, arg):
            if event in ["line"]: # , "call", "return"]:
                # Retrieve the filename and line number
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno

                # Retrieve the specific line of code
                line = linecache.getline(filename, lineno).rstrip()

                # Retrieve the local variables at this point in the code
                local_variables = frame.f_locals
                trace_info = f"Event: {event} | Line {lineno}: {line} | Local variables: {local_variables}\n"
                
                # Write the trace info to the file
                file.write(trace_info)

            return trace_function

        # Set the trace with a file-bound trace function
        sys.settrace(trace_function)

        # Call the function to be traced
        return_val = function(*args, **kwargs)
        file.write('RETURNED: '+str(return_val)+'\n')

        # Clear the trace
        sys.settrace(None)

def my_function():
    a = 10
    b = 20
    c = a + b
    print(c)

def solve(st):
    if len(st) < 2:
        return 0
    a = 10
    for i in range(len(st)-1, 0, -1):
        if st[:i] == st[-(len(st)-i):]:
            return i
    return 0

set_trace_to_file('trace_output.txt', solve, 'aaaa')
