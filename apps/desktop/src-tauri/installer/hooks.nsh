; Anime Pic Manage —— 安装器 / 卸载器自定义钩子
;
; 由 tauri.conf.json → bundle.windows.nsis.installerHooks 引入。Tauri 模板在
; 最前面 !include 本文件，因此这里可以：
;
;   * 声明卸载向导的自定义页面（页面顺序 = 声明顺序，所以它排在“确认卸载”之前）
;   * 实现 NSIS_HOOK_POSTUNINSTALL，在模板清理完程序本体后，按用户选择删除
;     安装目录内的数据（缓存 / 模型 / 产物）
;
; 两个必须记住的约定：
;
;   1. 本文件必须以 UTF-8 BOM 保存。makensis 对没有 BOM 的脚本按系统 ACP
;      （中文 Windows 上是 GBK）解码，中文会变成乱码。
;   2. “安装时自动沿用上次的安装目录”由 Tauri 模板自带的
;      RestorePreviousInstallLocation 完成：安装时把 $INSTDIR 写入
;      HKCU\Software\<厂商>\<产品名>，下次安装读回。这里不需要重复实现，
;      但 tauri.conf.json 必须配置 bundle.publisher，该注册表键才有意义。

Var APMCacheCheckbox
Var APMModelsCheckbox
Var APMOutputCheckbox
Var APMDeleteCache
Var APMDeleteModels
Var APMDeleteOutput

; 安装完成后补齐与便携包一致的目录结构：数据、模型、产物都落在软件目录里，
; 用户装完就能看到它们的落点，而不是等第一次用到时凭空出现。
!macro NSIS_HOOK_POSTINSTALL
  CreateDirectory "$INSTDIR\data"
  CreateDirectory "$INSTDIR\models"
  CreateDirectory "$INSTDIR\output\generated"
  CreateDirectory "$INSTDIR\output\loras"
  CreateDirectory "$INSTDIR\output\datasets"
  CreateDirectory "$INSTDIR\temp"
!macroend

; 卸载器页面：选择清理范围。声明位置在 MUI_UNPAGE_CONFIRM 之前，所以用户
; 先看到选项，再看到“确认卸载”。
UninstPage custom un.APMOptionsShow un.APMOptionsLeave

Function un.APMOptionsShow
  !insertmacro MUI_HEADER_TEXT "卸载选项" "选择是否删除安装目录里的缓存与模型"

  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}

  ${NSD_CreateLabel} 0 0 100% 26u "程序本体一定会被删除。下面这些文件夹在安装目录内，请选择是否一并删除："
  Pop $0

  ${NSD_CreateCheckbox} 0 30u 100% 24u "模型缓存 data\hf-cache —— 打标模型、参考匹配模型与相关缓存（约 700MB，可重新下载）"
  Pop $APMCacheCheckbox
  ${NSD_Check} $APMCacheCheckbox

  ${NSD_CreateCheckbox} 0 58u 100% 24u "识别模型 models\ —— 角色识别与头部检测模型（约 1.1GB，可重新下载）"
  Pop $APMModelsCheckbox
  ${NSD_Check} $APMModelsCheckbox

  ${NSD_CreateCheckbox} 0 86u 100% 24u "训练与出图产物 output\ —— LoRA、训练集、生成图（删掉无法恢复）"
  Pop $APMOutputCheckbox
  ${NSD_Uncheck} $APMOutputCheckbox

  ${NSD_CreateLabel} 0 116u 100% 30u "取消勾选的文件夹会留在安装目录里：数据库、人工矫正结果与设置都在 data\ 内，重新安装后可直接继续使用。"
  Pop $0

  nsDialogs::Show
FunctionEnd

Function un.APMOptionsLeave
  ${NSD_GetState} $APMCacheCheckbox $APMDeleteCache
  ${NSD_GetState} $APMModelsCheckbox $APMDeleteModels
  ${NSD_GetState} $APMOutputCheckbox $APMDeleteOutput
FunctionEnd

; 清理逻辑。静默卸载（/S 或 /P）不会显示上面的页面，此时默认保留全部数据，
; 只有显式传参才会删除，便于脚本化卸载时不会误删用户的训练产物：
;
;   uninstall.exe /S /DELCACHE /DELMODELS /DELOUTPUT
;
!macro NSIS_HOOK_POSTUNINSTALL
  ${GetOptions} $CMDLINE "/DELCACHE" $R0
  ${IfNot} ${Errors}
    StrCpy $APMDeleteCache 1
  ${EndIf}
  ${GetOptions} $CMDLINE "/DELMODELS" $R0
  ${IfNot} ${Errors}
    StrCpy $APMDeleteModels 1
  ${EndIf}
  ${GetOptions} $CMDLINE "/DELOUTPUT" $R0
  ${IfNot} ${Errors}
    StrCpy $APMDeleteOutput 1
  ${EndIf}

  ${If} $APMDeleteCache = 1
    DetailPrint "删除模型缓存 data\hf-cache"
    RMDir /r "$INSTDIR\data\hf-cache"
  ${EndIf}
  ${If} $APMDeleteModels = 1
    DetailPrint "删除识别模型 models"
    RMDir /r "$INSTDIR\models"
  ${EndIf}
  ${If} $APMDeleteOutput = 1
    DetailPrint "删除训练与出图产物 output"
    RMDir /r "$INSTDIR\output"
  ${EndIf}

  ; app\ 属于程序自带运行时（含一键下载的 CUDA 运行库），不属于用户数据，
  ; 一律清理，避免卸载后留下 1GB 以上的残留。
  RMDir /r "$INSTDIR\app"

  ; 数据都清掉后安装目录应当为空；若用户选择保留，这一步会安静地失败，
  ; 目录及其中的用户数据原样保留。
  RMDir "$INSTDIR"
!macroend
