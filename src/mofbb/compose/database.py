"""
Adapted from https://github.com/Sangwon91/PORMAKE/blob/main/src/pormake/database.py
"""
import pickle
from pathlib import Path

from mofbb.compose.building_block import BuildingBlock
from mofbb.compose.log import logger
from mofbb.compose.topology import Topology


class Database:
    """Load the bundled topology and building-block libraries by name.

    PORMAKE ships a ``database/`` directory containing abstract nets as
    ``.cgd`` files (cached as ``.pickle`` for fast reloading) and
    molecular building blocks as ``.xyz`` files. ``Database`` is the
    central registry that resolves names to fully constructed
    :class:`~pormake.topology.Topology` and
    :class:`~pormake.building_block.BuildingBlock` objects, so the rest
    of the assembly pipeline can request components by short name
    instead of file path.

    By default it points at the packaged ``database/topologies`` and
    ``database/bbs`` folders, but custom directories may be supplied to
    work with user-provided libraries.

    Parameters
    ----------
    topo_dir : str or pathlib.Path, optional
        Directory holding topology ``.cgd``/``.pickle`` files. If
        ``None``, the bundled ``database/topologies`` directory is used.
    bb_dir : str or pathlib.Path, optional
        Directory holding building-block ``.xyz`` files. If ``None``,
        the bundled ``database/bbs`` directory is used.

    Attributes
    ----------
    topo_dir : pathlib.Path
        Resolved directory of topology files.
    bb_dir : pathlib.Path
        Resolved directory of building-block files.

    Raises
    ------
    Exception
        If either resolved directory does not exist.
    """

    def __init__(self, topo_dir=None, bb_dir=None):
        """Resolve and validate the topology/building-block directories."""
        db_path = Path(__file__).parent / "database"
        self.topo_dir = Path(topo_dir or db_path / "topologies")
        self.bb_dir = Path(bb_dir or db_path / "bbs")
        for directory in (self.topo_dir, self.bb_dir):
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Database directory does not exist: {directory}"
                )

    @property
    def topology_list(self):
        """list of str: Names of all available topologies."""
        return [path.stem for path in self.topo_dir.glob("*.cgd")]

    @property
    def building_block_list(self):
        """list of str: Names of all available building blocks."""
        return [path.stem for path in self.bb_dir.glob("*.xyz")]

    def serialize(self):
        """Pre-pickle every bundled topology for fast later loading.

        Builds each topology from its ``.cgd`` file and writes a
        matching ``.pickle`` next to it. Because parsing a ``.cgd`` net
        is comparatively slow, pickling the libraries up front lets
        :meth:`get_topology` reload them quickly on subsequent runs.
        Topologies that fail to parse are skipped with a debug message.
        """
        print("Database serialization starts.")
        n_topos = len(self.topology_list)
        for i, name in enumerate(self.topology_list, start=1):
            cgd_path = self.topo_dir / (name + ".cgd")
            try:
                topo = Topology(cgd_path)
            except Exception as e:
                logger.debug(f'Invalid topology {e}')
                continue

            # Save pickle
            pickle_path = self.topo_dir / (name + ".pickle")
            with pickle_path.open("wb") as f:
                pickle.dump(topo, f)
            logger.debug("Pickle %s saved" % pickle_path)

            percent = i / n_topos * 100
            print("\rProgress: %.1f %% (%d/%d)" % (percent, i, n_topos), end="")

    def get_topology(self, name):
        """Load a topology by name, using the pickle cache when present.

        Attempts to load the pre-pickled topology first; if no pickle
        exists, the ``.cgd`` file is parsed into a
        :class:`~pormake.topology.Topology` and the result is cached as
        a ``.pickle`` for next time.

        Parameters
        ----------
        name : str
            Topology name without extension (e.g. ``"pcu"``).

        Returns
        -------
        pormake.topology.Topology
            The requested topology.

        Raises
        ------
        Exception
            If the topology cannot be loaded from pickle or ``.cgd``.
        """
        pickle_path = self.topo_dir / (name + ".pickle")
        if pickle_path.exists():
            with pickle_path.open("rb") as f:
                topology = pickle.load(f)
            logger.debug("Topology is loaded from pickle.")
            return topology

        topology = Topology(self.topo_dir / (name + ".cgd"))
        with pickle_path.open("wb") as f:
            pickle.dump(topology, f)
        logger.debug("Pickle %s saved" % pickle_path)
        return topology

    def get_building_block(self, name):
        """Load a building block by name from the ``.xyz`` library.

        Parameters
        ----------
        name : str
            Building-block name (with or without ``.xyz`` extension);
            only the stem is used to locate the file.

        Returns
        -------
        pormake.building_block.BuildingBlock
            The requested building block.

        Raises
        ------
        Exception
            If the building block cannot be loaded.
        """
        name = Path(name).stem + ".xyz"
        return BuildingBlock(self.bb_dir / name)
