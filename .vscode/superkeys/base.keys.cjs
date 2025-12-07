a("_run_task").c("workbench.action.tasks.runTask").b("ctrl+t").w();

a("_open_terminal").f().b("ctrl+space").w();

a("_reload_window").c("workbench.action.reloadWindow").b("ctrl+r").w();

a("_remove_all_comments").c("remove-comments.removeAllComments").b("alt+m").w();

a("_toggle_copilot").c("github.copilot.completions.toggle").b("ctrl+e").w();

a("_close_all_editors").c("workbench.action.closeAllEditors").b("ctrl+w").w();

a("_close_other_editors")
  .c("workbench.action.closeOtherEditors")
  .b("ctrl+q")
  .w();

a("_refresh")
  .c("_reload_configs")
  .c("search.action.clearSearchResults")
  .c("closeFindWidget")
  .c("workbench.files.action.collapseExplorerFolders")
  .c("workbench.action.closeAuxiliaryBar")
  .c("leaveEditorMessage")
  .c("workbench.files.action.focusFilesExplorer")
  .b("f5")
  .w();

a("_hard_refresh")
  .c("workbench.action.terminal.killAll")
  .c("workbench.action.tasks.terminate")
  .c("workbench.action.terminal.new")
  .c("_close_all_editors")
  .e("clear")
  .b("alt+f5")
  .w();

a("_run_view")
  .c("superkeys.runView")
  .c("editor.action.goToTopHover")
  .b("f11")
  .w();

a("_open_view").c("superkeys.openView").b("f10").w();

a("_reload_configs")
  .c("superkeys.loadConfigs")
  .e("Remove-Item -ErrorAction SilentlyContinue __dump.txt")
  .b("f12")
  .w();

a("_zoom_out").c("workbench.action.zoomOut").b("ctrl+[").w();

a("_zoom_in").c("workbench.action.zoomIn").b("ctrl+]").w();

a("_do_quicksave")
  .c("_open_terminal")
  .e("git add .; git commit -m 'quicksave'; git push")
  .b("alt+y")
  .w();
