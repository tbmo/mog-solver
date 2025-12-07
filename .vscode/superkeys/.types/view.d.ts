/**
 * Superkeys View API
 * Create file views to aggregate and filter workspace files
 */

declare class View {
  /**
   * Creates a new view with the specified name
   * @param name - The name of the view (must be unique)
   */
  constructor(name: string);


  /**
   * Include files matching glob patterns
   * @param pattern - Glob pattern or array of patterns (e.g., "*.js", "src/⁎⁎/*.ts")
   * @param beg - If true, prepends ⁎⁎/ to pattern (matches anywhere)
   * @param end - If true, appends /⁎⁎ to pattern (includes subdirectories)
   */
  ig(pattern: string | string[], beg?: boolean, end?: boolean): View;

  /**
   * Exclude files matching glob patterns
   * @param pattern - Glob pattern or array of patterns
   * @param beg - If true, prepends ⁎⁎/ to pattern
   * @param end - If true, appends /⁎⁎ to pattern
   */
  xg(pattern: string | string[], beg?: boolean, end?: boolean): View;

  /**
   * Include files from another view
   * @param view - Name of the view to include or array of view names
   */
  iv(view: string | string[]): View;

  /**
   * Exclude files from another view
   * @param view - Name of the view to exclude or array of view names
   */
  xv(view: string | string[]): View;

  /**
   * Include files containing specific content
   * @param pattern - String to search for, or regex pattern (e.g., "/TODO:/i")
   */
  ip(pattern: string | string[]): View;

  /**
   * Exclude files containing specific content
   * @param pattern - String to search for, or regex pattern
   */
  xp(pattern: string | string[]): View;

  /**
   * Register this view to make it available
   */
  w(): View;


  /**
   * Only generate the header/tree view without file contents
   */
  nobody(): View;

  /**
   * Add a prompt/description that will appear in the output
   * @param headerPrompt - Text to include as a prompt section
   */
  prompt(headerPrompt: string): View;

  /**
   * Mark this view as private (won't show in command palette)
   */
  private(): View;

  /**
   * Include debug information in the output
   */
  debug(): View;

  /**
   * Register this view to make it available
   * Must be called at the end of the chain
   */
  write(): View;
}

/**
 * Create a new view with the specified name (Keys API style)
 * @param name - The name of the view (must be unique)
 * @example
 * // Using short syntax
 * v("src")
 *   .ig("src/⁎⁎/*.js")
 *   .xg("*.test.js") 
 *   .nobody()
 *   .w();
 * 
 * // Using traditional syntax
 * new View("src")
 *   .includeGlob("src/⁎⁎/*.js")
 *   .excludeGlob("*.test.js")
 *   .nobody()
 *   .write();
 */
declare function v(name: string, inclusive?: boolean): View;

/**
 * Global View constructor available in view config files
 */
declare const View: typeof View;
