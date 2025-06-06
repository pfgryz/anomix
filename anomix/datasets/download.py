import asyncio
from pathlib import Path
from typing import Optional

import aiohttp
from tqdm import tqdm

from anomix.config import RAW_DATA_DIR
from anomix.datasets.definition import DatasetDefinition


async def download_file(session, url: str, destination_path: Path, progress_bar: Optional[tqdm]) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    async with session.get(url) as response:
        with open(destination_path, "wb") as handle:
            async for chunk in response.content.iter_chunked(1024):
                if not chunk:
                    break

                handle.write(chunk)

        if progress_bar is not None:
            progress_bar.update(1)


async def download_datasets(definitions: list[DatasetDefinition], num_workers: int = 8) -> None:
    async with aiohttp.ClientSession() as session:
        progress_bar = tqdm(total=len(definitions), desc="Downloading Datasets")
        progress_bar.update(0)

        to_download = []
        for definition in definitions:
            format = definition.format if definition.archive is None else definition.archive
            destination_path = RAW_DATA_DIR / f"{definition.name}.{format}"

            if not destination_path.exists():
                to_download.append((definition.url, destination_path))
            else:
                progress_bar.update(1)

        semaphore = asyncio.Semaphore(num_workers)

        async def download_worker(url, destination_path):
            async with semaphore:
                await download_file(session, url, destination_path, progress_bar)

        await asyncio.gather(*(download_worker(url, destination_path) for url, destination_path in to_download))

        progress_bar.close()
