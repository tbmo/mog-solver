v("folders", true).nobody().write();
v("keys_config").ig("**/.vscode/superkeys/*.keys.cjs").write();
v("views_config").ig("**/.vscode/superkeys/*.views.cjs").write();
v("settings").ig("**/.vscode/settings.json").write();

v("code").ig("main.py").ig("config.yaml").write();
v("shaders").ig("**/shaders/**").write();

v("source").iv("code").iv("shaders").write();