{
  description = "Themis - an assistent in finding fitting feedback items";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    devshell = {
      url = "github:numtide/devshell";
      inputs = {
        flake-utils.follows = "flake-utils";
        nixpkgs.follows = "nixpkgs";
      };
    };
  };

  outputs = { self, nixpkgs, devshell, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; overlays = [ devshell.overlays.default ]; };
        tree-sitter = pkgs.python3Packages.buildPythonPackage rec {
          pname = "tree_sitter";
          version = "0.20.0";
          format = "setuptools";

          src = pkgs.python3Packages.fetchPypi {
            inherit pname version;
            sha256 = "GUD2S+HoycPA40oiWPHkwyQgdTTVse78WrKWCp2Y9mg=";
          };

          pythonImportsCheck = [ "tree_sitter" ];
        };
        bounded-pool-executor = pkgs.python3Packages.buildPythonPackage rec {
          pname = "bounded_pool_executor";
          version = "0.0.3";
          format = "setuptools";

          src = pkgs.python3Packages.fetchPypi {
            inherit pname version;
            sha256 = "4JIiG8OK3lVeEGSDH57YAFgPo0pLbY6d082WFUlif24=";
          };

          propagatedBuildInputs = [];

          pythonImportsCheck = [ "bounded_pool_executor" ];
        };
        pqdm = pkgs.python3Packages.buildPythonPackage rec {
          pname = "pqdm";
          version = "0.2.0";
          format = "setuptools";

          src = pkgs.python3Packages.fetchPypi {
            inherit pname version;
            sha256 = "2Z0B/kmNMntEDr/gjBTITg3J7M5hcu+aMflrsar06eM=";
          };

          doCheck = false;

          propagatedBuildInputs = [
            pkgs.python3Packages.typing-extensions
            pkgs.python3Packages.tqdm
            bounded-pool-executor
          ];

          pythonImportsCheck = [ "pqdm.processes" "pqdm.threads" ];
        };
      in
      {
        devShells = rec {
          default = themis;
          themis = pkgs.devshell.mkShell {
            name = "Themis";
            imports = [ "${devshell}/extra/language/c.nix" ];
            packages = with pkgs; [
              nixpkgs-fmt
              (python3.withPackages (ps: [ tree-sitter ps.pylint ps.tqdm ps.python-lsp-server pqdm ]))
            ];
            language.c = {
              compiler = pkgs.gcc;
            };
          };
        };
      }
    );
}
