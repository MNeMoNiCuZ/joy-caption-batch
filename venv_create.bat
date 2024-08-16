@echo off
setlocal enabledelayedexpansion

:: Initialize counter
set COUNT=0

:: Directly parse the output of `py -0p` to get versions and their paths
for /f "tokens=1,*" %%a in ('py -0p') do (
    :: Filter lines that start with a dash, indicating a Python version, and capture the path
    echo %%a | findstr /R "^[ ]*-" > nul && (
        set /a COUNT+=1
        set "pythonVersion=%%a"
        :: a quick, dirty but understandable solution
        set "pythonVersion=!pythonVersion:-32=!"
        set "pythonVersion=!pythonVersion:-64=!"
        set "pythonVersion=!pythonVersion:-=!"
        set "pythonVersion=!pythonVersion:V:=!"
        set "PYTHON_VER_!COUNT!=!pythonVersion!"
        set "PYTHON_PATH_!COUNT!=%%b"  :: Store the path in a separate variable
        echo !COUNT!. %%a at %%b
    )
)

:: Make sure at least one Python version was found
if %COUNT%==0 (
    echo No Python installations found via Python Launcher. Exiting.
    goto end
)

:: Prompt user to select a Python version (default is 1)
set /p PYTHON_SELECTION="Select a Python version by number (Press Enter for default = '1'): "
if "!PYTHON_SELECTION!"=="" set PYTHON_SELECTION=1

:: Extract the selected Python version tag and parse the version number more accurately
set SELECTED_PYTHON_VER=!PYTHON_VER_%PYTHON_SELECTION%!

:: Confirm the selected Python version
echo Using Python version %SELECTED_PYTHON_VER%

:: Prompt for virtual environment name with default 'venv'
set VENV_NAME=venv
set /p VENV_NAME="Enter the name for your virtual environment (Press Enter for default 'venv'): "
if "!VENV_NAME!"=="" set VENV_NAME=venv

:: Create the virtual environment using the selected Python version
echo Creating virtual environment named %VENV_NAME%...
py -%SELECTED_PYTHON_VER% -m venv %VENV_NAME%

:: Add .gitignore to the virtual environment folder
echo Creating .gitignore in the %VENV_NAME% folder...
(
echo # Ignore all content in the virtual environment directory
echo *
echo # Except this file
echo !.gitignore
) > %VENV_NAME%\.gitignore

:: Generate the venv_activate.bat file
echo Generating venv_activate.bat...
(
echo @echo off
echo cd %%~dp0
echo set VENV_PATH=%VENV_NAME%
echo.
echo echo Activating virtual environment...
echo call "%%VENV_PATH%%\Scripts\activate"
echo echo Virtual environment activated.
echo cmd /k
) > venv_activate.bat

:: Generate the venv_update.bat file for a one-time pip upgrade
echo Generating venv_update.bat for a one-time pip upgrade...
(
echo @echo off
echo cd %%~dp0
echo echo Activating virtual environment %VENV_NAME% and upgrading pip...
echo call "%VENV_NAME%\Scripts\activate"
echo python -m pip install --upgrade pip
echo echo Pip has been upgraded in the virtual environment %VENV_NAME%.
echo echo To deactivate, manually type 'deactivate'.
echo call "%VENV_NAME%\Scripts\activate"
) > venv_update.bat

:: Ask the user if they want to upgrade pip now
echo.
set /p UPGRADE_NOW="Do you want to upgrade pip now? (Y/N) (Press Enter for default 'Y'): "
if not defined UPGRADE_NOW set UPGRADE_NOW=Y
if /I "%UPGRADE_NOW%"=="Y" (
    echo Upgrading pip and activating the virtual environment...
    call venv_update.bat
)

:: Ensure we are in the virtual environment
echo.
call "%VENV_NAME%\Scripts\activate"

:: Ask the user if they want to install uv
echo.
set INSTALL_UV=Y
echo uv pip package installer:
echo uv is a Python package that provides enhanced capabilities for managing Python environments and dependencies.
set /p INSTALL_UV="Do you want to install 'uv' package? (Y/N) (Press Enter for default 'Y'): "
if "!INSTALL_UV!"=="" set INSTALL_UV=Y
set INSTALL_UV=!INSTALL_UV:~0,1!

if /I "!INSTALL_UV!"=="Y" (
    echo Installing 'uv' package...
    pip install uv
    set UV_INSTALLED=1
) else (
    set UV_INSTALLED=0
)

:: Check if requirements.txt exists
if exist requirements.txt (
    echo requirements.txt found.
    :: Ask the user if they want to install requirements
    if "!UV_INSTALLED!"=="1" (
        set /p INSTALL_REQUIREMENTS="Do you wish to run 'uv pip install -r requirements.txt'? (Y/N) (Press Enter for default 'Y'): "
    ) else (
        set /p INSTALL_REQUIREMENTS="Do you wish to run 'pip install -r requirements.txt'? (Y/N) (Press Enter for default 'Y'): "
    )
    if "!INSTALL_REQUIREMENTS!"=="" set INSTALL_REQUIREMENTS=Y
    echo User selected: !INSTALL_REQUIREMENTS!
    if /I "!INSTALL_REQUIREMENTS!"=="Y" (
        echo Installing requirements from requirements.txt...
        call "%VENV_NAME%\Scripts\activate"  :: Ensure the virtual environment is active
        if "!UV_INSTALLED!"=="1" (
            uv pip install -r requirements.txt
        ) else (
            pip install -r requirements.txt
        )
        echo Requirements installed.
    ) else (
        echo Skipping requirements installation.
    )
) else (
    echo requirements.txt not found. Skipping requirements installation.
)

:: List installed packages
echo Listing installed packages...
pip list

echo Setup complete. Your virtual environment is ready.
echo To deactivate the virtual environment, type 'deactivate'.

:: Keep the command prompt open
cmd /k

:: After completing the main part of the script, jump to cleanup
goto cleanup

:cleanup
:: Clean up
echo Cleanup complete.

:: End of script
endlocal
