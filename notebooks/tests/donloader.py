""" Video downloader from linkedin"""
import os
import subprocess
from typing import Optional

def download_video_from_linkedin(url: str, output_path: Optional[str] = None) -> str:
    """Downloads a video from LinkedIn using yt-dlp.

    Args:
        url (str): The URL of the LinkedIn video to download.
        output_path (Optional[str]): The path where the downloaded video will be saved. 
                                     If None, the video will be saved in the current directory.

    Returns:
        str: The path to the downloaded video file.
    """
    if output_path is None:
        output_path = os.getcwd()
    
    # Ensure yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        raise RuntimeError("yt-dlp is not installed. Please install it using 'pip install yt-dlp'.")

    # Download the video
    try:
        result = subprocess.run(
            ["yt-dlp", url, "-o", os.path.join(output_path, "%(title)s.%(ext)s")],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode())
        raise RuntimeError("Failed to download the video. Please check the URL and try again.")

    # Return the path to the downloaded video
    return os.path.join(output_path, "%(title)s.%(ext)s")


download_video_from_linkedin("https://www.linkedin.com/feed/update/urn:li:groupPost:121615-7460963486890885123?utm_source=share&utm_medium=member_desktop&rcm=ACoAADC0VPcB_xGRX6HLRbpikRLfbNt9xWZSlco")
