#include <stdio.h>
#include <stdlib.h>

#include "error_handler.h"

void handle_error (const char* message) {
  fprintf(stderr, "Unfortunately, an error has occurred!\n%s", message);
  exit(EXIT_FAILURE);
}

void handle_verbose_error (const char* file_name, const char* function_name, int line_number, const char* message) {
  fprintf(stderr, "Unfortunately, an error has occurred!\n"
                  "    File: %s\n"
                  "Function: %s\n"
                  "    Line: %d\n"
                  " Message: %s\n",
    file_name, function_name, line_number, message
  );
  exit(EXIT_FAILURE);
}
