v("folders", true).nobody().write();
v("keys_config").ig("**/.vscode/superkeys/*.keys.cjs").write();
v("views_config").ig("**/.vscode/superkeys/*.views.cjs").write();
v("settings").ig("**/.vscode/settings.json").write();

v("shaders").ig("**/shaders/**").write();