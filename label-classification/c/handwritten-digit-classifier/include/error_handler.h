#ifndef HDC_ERROR_HANDLER_H
#define HDC_ERROR_HANDLER_H

#define error(message) handle_verbose_error(__FILE__, __func__, __LINE__, message)

void handle_error (const char*);
void handle_verbose_error (const char*, const char*, int, const char*);

#endif // HDC_ERROR_HANDLER_H
