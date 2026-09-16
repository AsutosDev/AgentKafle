; NSIS installer script for AgentKafle v1.0.0
; Run on Windows: makensis installer.nsi

!define APP_NAME "AgentKafle"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "AgentKafle"
!define APP_EXE "AgentKafle.exe"
!define APP_DIR "AgentKafle"

RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\${APP_DIR}"
InstallDirRegKey HKCU "Software\${APP_NAME}" ""

Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

Name "${APP_NAME} ${APP_VERSION}"
OutFile "AgentKafle-v${APP_VERSION}-Windows-Installer.exe"
InstallDirRegKey HKCU "Software\${APP_NAME}" ""
ShowInstDetails show

Section "MainSection" SEC01
    SetOutPath "$INSTDIR"
    File /r "dist\${APP_DIR}\*"
    
    WriteRegStr HKCU "Software\${APP_NAME}" "" $INSTDIR
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
        "DisplayName" "${APP_NAME}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
        "UninstallString" "$INSTDIR\uninstall.exe"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
        "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
        "Publisher" "${APP_PUBLISHER}"
    
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
    CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}"
    
    WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\${APP_EXE}"
    Delete "$INSTDIR\uninstall.exe"
    RMDir /r "$INSTDIR"
    
    Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
    Delete "$DESKTOP\${APP_NAME}.lnk"
    RMDir "$SMPROGRAMS\${APP_NAME}"
    
    DeleteRegKey HKCU "Software\${APP_NAME}"
    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
SectionEnd
