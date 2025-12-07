/**
 * Superkeys Keybinding API
 * Define custom keybindings and command sequences
 */

declare class Action {
  /**
   * Creates a new action/keybinding rule
   * @param name - Unique name for this action (prefix with _ for private actions)
   */
  constructor(name: string);

  /**
   * Add a VS Code command to the action sequence
   * @param cmd - Command ID or another Action name
   * @param args - Optional arguments for the command
   */
  command(cmd: string | Action, args?: any): Action;
  
  /**
   * Shorthand for command()
   */
  c(cmd: string | Action, args?: any): Action;

  /**
   * Open a file or folder in the editor
   * @param path - Absolute or relative path to the file or folder
   */
  open(path: string): Action;

  /**
   * Shorthand for open()
   */
  o(path: string): Action;

  /**
   * Execute a shell command in the terminal
   * @param shell - Shell command to execute
   */
  execute(shell: string): Action;
  
  /**
   * Shorthand for execute()
   */
  e(shell: string): Action;

  /**
   * Focus the terminal
   */
  focus(): Action;
  
  /**
   * Shorthand for focus()
   */
  f(): Action;

  /**
   * Bind this action to a key combination
   * @param key - Key combination (e.g., "ctrl+shift+p", "f5", "alt+m")
   * @param exclusions - Array of command names to keep active for this key
   * @example
   * // Take over F2 key but keep renameFile active
   * a("test").bind("f2", ["renameFile"])
   * 
   * // Take over F2 but keep renameFile active when certain conditions are met
   * a("test").bind("f2", ["renameFile", "commandName"])
   */
  bind(key: string, exclusions?: string[]): Action;
  
  /**
   * Shorthand for bind()
   */
  b(key: string, exclusions?: string[]): Action;

  /**
   * Register this action (must be called at the end)
   */
  write(): string;
  
  /**
   * Shorthand for write()
   */
  w(): string;
}

/**
 * Create a new action/keybinding rule
 * @param name - Unique name for this action
 * @example
 * a("_my_command")
 *   .c("workbench.action.files.save")
 *   .c("editor.action.formatDocument")
 *   .b("ctrl+s")
 *   .w();
 */
declare function a(name: string): Action;

/**
 * Available VS Code variables for use in commands:
 * - ${workspaceFolder} - The path of the workspace folder
 * - ${file} - The current opened file
 * - ${fileBasename} - The current opened file's basename
 * - ${fileDirname} - The current opened file's dirname
 * - ${fileExtname} - The current opened file's extension
 * - ${lineNumber} - The current selected line number
 * - ${selectedText} - The current selected text
 */
