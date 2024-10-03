import os
import server
from server import PromptServer
from aiohttp import web
import subprocess
import folder_paths
import asyncio
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VideoViewerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "view_video"
    CATEGORY = "video"
    OUTPUT_NODE = True

    def view_video(self, video_path):
        return {}

    @classmethod
    def IS_CHANGED(s, video_path):
        return float("nan")  # Always return a new value

# Register the route for video streaming
@server.PromptServer.instance.routes.get("/view_video")
async def view_video(request):
    try:
        query = request.rel_url.query
        if "video_path" not in query:
            return web.Response(status=400, text="No video path provided")
        video_path = query["video_path"]

        logger.debug(f"Requested video path: {video_path}")

        # Check if the file exists
        if not os.path.isfile(video_path):
            logger.error(f"Video file not found: {video_path}")
            return web.Response(status=404, text="Video file not found")

        # Use ffprobe to get video information
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=codec_name,width,height",
            "-of", "json",
            video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe_result.returncode != 0:
            logger.error(f"ffprobe failed: {probe_result.stderr}")
            return web.Response(status=500, text="Failed to probe video file")

        logger.debug(f"ffprobe result: {probe_result.stdout}")

        # Prepare ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-c:v", "libvpx-vp9",
            "-crf", "30",
            "-b:v", "0",
            "-b:a", "128k",
            "-c:a", "libopus",
            "-f", "webm",
            "pipe:1"
        ]

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        # Start ffmpeg process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        response = web.StreamResponse()
        response.content_type = 'video/webm'

        await response.prepare(request)

        try:
            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    break
                await response.write(chunk)

            await response.write_eof()
            return response

        except Exception as e:
            logger.error(f"Error during video streaming: {str(e)}")
            return web.Response(status=500, text=f"Error during video streaming: {str(e)}")

        finally:
            if process:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg process did not terminate, killing it")
                    process.kill()

    except Exception as e:
        logger.error(f"Unexpected error in view_video: {str(e)}")
        return web.Response(status=500, text=f"Unexpected error: {str(e)}")

# New route for handling video uploads
@server.PromptServer.instance.routes.post("/upload_video")
async def upload_video(request):
    try:
        reader = await request.multipart()
        field = await reader.next()
        if field.name != "video":
            return web.Response(status=400, text="Invalid form field")

        filename = field.filename
        if not filename:
            return web.Response(status=400, text="No filename provided")

        # Generate a unique filename to prevent overwriting
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(folder_paths.get_input_directory(), unique_filename)

        size = 0
        with open(file_path, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)

        logger.info(f"Video uploaded: {file_path}, size: {size} bytes")
        return web.json_response({"success": True, "filename": file_path, "size": size})

    except Exception as e:
        logger.error(f"Error during video upload: {str(e)}")
        return web.Response(status=500, text=f"Error during upload: {str(e)}")

